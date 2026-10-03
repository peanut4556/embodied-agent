"""Same-state one-step expert targets and projected finite-difference feasibility audit."""

import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", "outputs/hf-cache")

import mujoco
import numpy as np

from embodied_agent.correction_data import payload_digest
from embodied_agent.correction_finetune import correction_sequences
from embodied_agent.memory_evaluation import verify_physics
from embodied_agent.memory_policy import MemoryPolicy, digest
from embodied_agent.physics import PhysicsWorld
from scripts.audit_action_chain import shadow_step


def loss(next_state, target, ranges):
    return float(np.mean(((next_state - target) / ranges) ** 2))


def improve(model, data, action, target, bounds, limit, steps):
    """One frozen projected step; simulator derivatives, not policy gradients."""
    ranges = bounds[:, 1] - bounds[:, 0]
    low = np.maximum(bounds[:, 0], data.ctrl - limit)
    high = np.minimum(bounds[:, 1], data.ctrl + limit)
    action = np.clip(action, low, high)
    gradient = np.zeros(5)
    for j in range(5):
        plus, minus = action.copy(), action.copy()
        plus[j] = min(high[j], action[j] + 0.001 * ranges[j])
        minus[j] = max(low[j], action[j] - 0.001 * ranges[j])
        span = (plus[j] - minus[j]) / ranges[j]
        if span:
            gradient[j] = (
                loss(shadow_step(model, data, plus, steps), target, ranges)
                - loss(shadow_step(model, data, minus, steps), target, ranges)
            ) / span
    updated = np.clip(action - 0.25 * gradient * ranges, low, high)
    return updated, gradient


def audit(root, requested_groups=("train", "validation")):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    source = Path("outputs/datasets/corrections-early-v1")
    model_root = Path("outputs/models/joint-training-control/epoch-300")
    policy = MemoryPolicy(model_root)
    verify_physics(policy)
    manifest = json.loads((source / "recording.json").read_text())
    source_hash = payload_digest(source)
    # This loader enforces successful-only, validated source and expert-only masks.
    if not requested_groups or any(g not in ("train", "validation") for g in requested_groups):
        raise ValueError("development groups required")
    groups = {g: correction_sequences(source, g) for g in requested_groups}
    dataset = LeRobotDataset(manifest["repo_id"], root=source, video_backend="pyav")
    exact = dataset.hf_dataset.data.column("replay.action")
    offsets = np.cumsum([0] + [e["frames"] for e in manifest["episodes"]])
    root.mkdir(parents=True)
    report = {
        "status": "running",
        "weights_sha256": policy.metadata["weights_sha256"],
        "source_sha256": source_hash,
        "script_sha256": digest(__file__),
        "shadow_source_sha256": digest(Path(__file__).with_name("audit_action_chain.py")),
        "design": {
            "expert_sample_stride": 25,
            "finite_difference_range_fraction": 0.001,
            "normalized_gradient_step": 0.25,
            "horizon_control_periods": 1,
            "input_history": "recorded learner prefix then expert observations; not autonomous model rollout",
        },
        "test_executed": False,
        "model_retrained": False,
        "rows": [],
    }
    try:
        for group, episodes in groups.items():
            for sequence in episodes:
                ep = sequence["episode"]
                info = manifest["episodes"][ep]
                offset = int(offsets[ep])
                world = PhysicsWorld()
                try:
                    for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                        getattr(world.data, key)[:] = info["initial_state"][key]
                    world.data.time = info["initial_state"]["time"]
                    session = policy.session()
                    ranges = policy.bounds[:, 1] - policy.bounds[:, 0]
                    rate = np.array([2.0, 2.0, 2.0, 1.0, 1.0]) / policy.metadata["fps"]
                    steps = round(1 / policy.metadata["fps"] / world.model.opt.timestep)
                    start = int(np.flatnonzero(sequence["loss_mask"])[0])
                    for t, x in enumerate(sequence["x"]):
                        mujoco.mj_forward(world.model, world.data)
                        if not np.allclose(world.data.qpos[:5], x[6:11], atol=1e-6, rtol=0):
                            raise ValueError("recorded state replay diverged")
                        prediction = session.predict(x[None])[0]
                        expert = np.asarray(exact[offset + t].as_py(), dtype=float)
                        sample = sequence["loss_mask"][t] and (t - start) % 25 == 0
                        entry = None
                        if sample:
                            target = shadow_step(world.model, world.data, expert, steps)
                            action = world.data.ctrl + np.clip(
                                prediction - world.data.ctrl, -rate, rate
                            )
                            action = np.clip(action, policy.bounds[:, 0], policy.bounds[:, 1])
                            original = shadow_step(world.model, world.data, action, steps)
                            updated, gradient = improve(
                                world.model, world.data, action, target, policy.bounds, rate, steps
                            )
                            improved = shadow_step(world.model, world.data, updated, steps)
                            entry = {
                                "group": group,
                                "episode": ep,
                                "tick": t,
                                "loss_before": loss(original, target, ranges),
                                "loss_after": loss(improved, target, ranges),
                                "arm_state_rmse_rad": float(
                                    np.sqrt(np.mean((original[:3] - target[:3]) ** 2))
                                ),
                                "finger_state_rmse_m": float(
                                    np.sqrt(np.mean((original[3:] - target[3:]) ** 2))
                                ),
                                "model_command": action.tolist(),
                                "expert_command": expert.tolist(),
                                "projected_command": updated.tolist(),
                                "gradient": gradient.tolist(),
                                "model_next_joints": original.tolist(),
                                "expert_next_joints": target.tolist(),
                            }
                        world.data.ctrl[:] = expert
                        for _ in range(steps):
                            mujoco.mj_step(world.model, world.data)
                        if entry:
                            error = float(np.max(np.abs(world.data.qpos[:5] - target)))
                            if error > 1e-10:
                                raise ValueError("shadow expert target differs from actual replay")
                            entry["target_replay_max_error"] = error
                            report["rows"].append(entry)
                finally:
                    world.close()
                print(
                    json.dumps(
                        {
                            "group": group,
                            "episode": ep,
                            "sampled": sum(r["episode"] == ep for r in report["rows"]),
                        }
                    ),
                    flush=True,
                )
        if payload_digest(source) != source_hash:
            raise ValueError("source changed")
        report["scores"] = {}
        for group in groups:
            rows = [r for r in report["rows"] if r["group"] == group]
            report["scores"][group] = {
                "samples": len(rows),
                "mean_loss_before": float(np.mean([r["loss_before"] for r in rows])),
                "mean_loss_after": float(np.mean([r["loss_after"] for r in rows])),
                "improved": sum(r["loss_after"] < r["loss_before"] - 1e-12 for r in rows),
                "worsened": sum(r["loss_after"] > r["loss_before"] + 1e-12 for r in rows),
                "max_target_replay_error": max(r["target_replay_max_error"] for r in rows),
            }
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    audit("outputs/evaluations/state-targets-v1")
