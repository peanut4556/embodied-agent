"""Matched on-policy takeover branches with joint, object and contact future targets."""

import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", "outputs/hf-cache")

import mujoco
import numpy as np

from embodied_agent.correction_data import check_layout, payload_digest
from embodied_agent.memory_evaluation import verify_physics
from embodied_agent.memory_policy import MemoryPolicy, digest
from embodied_agent.physics import PhysicsWorld
from embodied_agent.reactive_evaluation import run_case

HORIZONS = (1, 25, 100, 200, 300)


def verify_takeover(row, takeover):
    for trace_key, record_key in [
        ("simulation_qpos", "qpos"),
        ("simulation_qvel", "qvel"),
        ("previous_command", "ctrl"),
    ]:
        if not np.allclose(row[trace_key], takeover[record_key], atol=1e-8, rtol=0):
            raise ValueError("autonomous continuation does not match recorded takeover")


def target_row(group, ep, horizon, initial, expert, autonomous):
    return {
        "group": group,
        "episode": ep,
        "horizon_ticks": horizon,
        "expert_joint_target": expert["joints"],
        "expert_object_target_m": expert["block_xyz"],
        "expert_holding_target": expert["holding"],
        "expert_inside_box": expert["inside_box"],
        "autonomous_joints": autonomous["joints"],
        "autonomous_object_m": autonomous["block_xyz"],
        "autonomous_holding": autonomous["holding"],
        "object_target_displacement_m": float(
            np.linalg.norm(np.array(expert["block_xyz"]) - initial)
        ),
        "object_continuation_error_m": float(
            np.linalg.norm(np.array(expert["block_xyz"]) - autonomous["block_xyz"])
        ),
        "arm_continuation_rmse_rad": float(
            np.sqrt(np.mean((np.array(expert["joints"][:3]) - autonomous["joints"][:3]) ** 2))
        ),
    }


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output = Path("outputs/evaluations/multistep-targets-v2")
    if output.exists():
        raise FileExistsError(output)
    source = Path("outputs/datasets/corrections-onpolicy-v1")
    manifest = json.loads((source / "recording.json").read_text())
    selection = json.loads((source / "selection.json").read_text())
    source_hash = payload_digest(source)
    if (
        manifest["status"] != "validated"
        or not selection["usable"]
        or selection["payload_sha256"] != source_hash
    ):
        raise ValueError("validated unchanged source required")
    check_layout(manifest)
    policy = MemoryPolicy("outputs/models/state-distill-distilled/epoch-300")
    verify_physics(policy)
    if (
        manifest["learner_weights_sha256"] != policy.metadata["weights_sha256"]
        or manifest["fps"] != 25
    ):
        raise ValueError("wrong rollout learner or clock")
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "source_sha256": source_hash,
        "weights_sha256": policy.metadata["weights_sha256"],
        "script_sha256": digest(__file__),
        "evaluator_sha256": digest(Path("src/embodied_agent/reactive_evaluation.py")),
        "horizons_ticks": HORIZONS,
        "fps": 25,
        "test_executed": False,
        "model_retrained": False,
        "baselines": [],
        "episodes": [],
        "targets": [],
    }
    try:
        baselines = {}
        for ep in manifest["episodes"]:
            x = ep["scenario"]["x"]
            if x in baselines:
                continue
            trace = []
            result, _ = run_case(
                policy,
                {"name": f"onpolicy-{round(x * 1000)}", "x": x},
                "memory",
                24,
                trace=trace,
                trace_until_tick=400,
            )
            baselines[x] = trace
            path = output / f"baseline-{round(x * 1000)}.json"
            path.write_text(json.dumps(trace) + "\n")
            report["baselines"].append({"x": x, "result": result, "trace_sha256": digest(path)})
            print(json.dumps({"baseline": x, "quality": result["quality"]}), flush=True)
        dataset = LeRobotDataset(manifest["repo_id"], root=source, video_backend="pyav")
        exact = dataset.hf_dataset.data.column("replay.action")
        offsets = np.cumsum([0] + [e["frames"] for e in manifest["episodes"]])
        for group in ("train", "validation"):
            for ep in manifest["split"][group]:
                record = manifest["episodes"][ep]
                tick = record["takeover"]["tick"]
                autonomous = baselines[record["scenario"]["x"]]
                verify_takeover(autonomous[tick], record["takeover"])
                report["episodes"].append(
                    {
                        "group": group,
                        "episode": ep,
                        "recoverable": record["success"],
                        "reason": record["reason"],
                        "takeover_matched": True,
                    }
                )
                if not record["success"]:
                    continue
                world = PhysicsWorld()
                expert = []
                try:
                    for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                        getattr(world.data, key)[:] = record["initial_state"][key]
                    world.data.time = record["initial_state"]["time"]
                    for t in range(record["frames"]):
                        mujoco.mj_forward(world.model, world.data)
                        observed = dataset[int(offsets[ep]) + t]["observation.state"].numpy()
                        if not np.allclose(world.data.qpos[:5], observed, atol=1e-6, rtol=0):
                            raise ValueError("expert replay state mismatch")
                        expert.append(
                            {
                                "joints": world.data.qpos[:5].tolist(),
                                "block_xyz": world.data.body("red_block").xpos.tolist(),
                                "holding": bool(world.holding),
                                "inside_box": bool(world.inside_box()),
                            }
                        )
                        world.data.ctrl[:] = np.asarray(exact[int(offsets[ep]) + t].as_py())
                        for _ in range(round(1 / 25 / world.model.opt.timestep)):
                            mujoco.mj_step(world.model, world.data)
                finally:
                    world.close()
                for horizon in HORIZONS:
                    end = tick + horizon
                    if end >= len(expert) or end >= len(autonomous):
                        raise ValueError("frozen target horizon exceeds trajectory")
                    report["targets"].append(
                        target_row(
                            group,
                            ep,
                            horizon,
                            np.array(expert[tick]["block_xyz"]),
                            expert[end],
                            autonomous[end],
                        )
                    )
        if payload_digest(source) != source_hash:
            raise ValueError("source changed")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    for group in ("train", "validation"):
        (output / f"{group}-targets.json").write_text(
            json.dumps(
                {
                    "source_sha256": source_hash,
                    "group": group,
                    "targets": [r for r in report["targets"] if r["group"] == group],
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
