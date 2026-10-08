"""Fixed camera translations, image occlusion and depth perturbations; no action trials."""

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "outputs/hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import mujoco
import numpy as np

from embodied_agent.correction_data import payload_digest
from embodied_agent.filtered_envelope import estimate
from embodied_agent.memory_policy import digest
from embodied_agent.physics import PhysicsWorld


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path("outputs/datasets/corrections-contact-oriented-v1")
    output = Path("outputs/evaluations/envelope-stress-v1")
    if output.exists():
        raise FileExistsError(output)
    m = json.loads((root / "recording.json").read_text())
    source = payload_digest(root)
    if (
        m["status"] != "validated"
        or json.loads((root / "selection.json").read_text())["payload_sha256"] != source
    ):
        raise ValueError("changed dataset")
    dataset = LeRobotDataset(m["repo_id"], root=root, video_backend="pyav")
    exact = dataset.hf_dataset.data.column("replay.action")
    report = {
        "status": "running",
        "source_sha256": source,
        "script_sha256": digest(__file__),
        "estimator_sha256": digest("src/embodied_agent/filtered_envelope.py"),
        "group": "train",
        "training_performed": False,
        "test_executed": False,
        "cases": [],
        "inputs": [],
    }
    output.mkdir(parents=True)
    try:
        for ep, wait in ((3, 0), (5, 100)):
            world = PhysicsWorld()
            renderer = None
            try:
                record = m["episodes"][ep]
                tick = record["takeover"]["tick"] + 74
                offset = sum(e["frames"] for e in m["episodes"][:ep])
                for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                    getattr(world.data, key)[:] = record["initial_state"][key]
                world.data.time = record["initial_state"]["time"]
                for t in range(tick + 1):
                    mujoco.mj_forward(world.model, world.data)
                    if not np.allclose(
                        world.data.qpos[:5],
                        dataset[offset + t]["observation.state"].numpy(),
                        atol=1e-6,
                        rtol=0,
                    ):
                        raise ValueError("replay mismatch")
                    if t == tick:
                        break
                    world.data.ctrl[:] = exact[offset + t].as_py()
                    for _ in range(20):
                        mujoco.mj_step(world.model, world.data)
                for _ in range(wait * 20):
                    mujoco.mj_step(world.model, world.data)
                mujoco.mj_forward(world.model, world.data)
                center = world.data.body("red_block").xpos.copy()
                rotation = world.data.body("red_block").xmat.reshape(3, 3)
                corners = (
                    np.array(
                        [
                            [a, b, c]
                            for a in (-0.025, 0.025)
                            for b in (-0.025, 0.025)
                            for c in (-0.025, 0.025)
                        ]
                    )
                    @ rotation.T
                    + center
                )
                truth_low, truth_high = corners.min(0), corners.max(0)
                truth_gap = 0.54 - truth_high[0]
                renderer = mujoco.Renderer(world.model, height=480, width=640)
                cam = world.model.camera("perception").id
                original = world.model.cam_pos[cam].copy()
                for shift in (-0.08, 0.0, 0.08):
                    world.model.cam_pos[cam] = original + [shift, 0, 0]
                    mujoco.mj_forward(world.model, world.data)
                    renderer.update_scene(world.data, camera="perception")
                    rgb = renderer.render().copy()
                    renderer.enable_depth_rendering()
                    depth = renderer.render().copy()
                    renderer.disable_depth_rendering()
                    pos, rot = world.data.cam_xpos[cam].copy(), world.data.cam_xmat[cam].copy()
                    fovy = world.model.cam_fovy[cam]
                    file = output / f"episode-{ep}-camera-{shift:+.2f}.npz"
                    np.savez_compressed(
                        file, rgb=rgb, depth=depth, position=pos, rotation=rot, fovy=fovy
                    )
                    report["inputs"].append({"path": str(file), "sha256": digest(file)})
                    r, g, b = rgb.astype(float).transpose(2, 0, 1)
                    red = (r > 70) & (r > 1.6 * g) & (r > 1.4 * b)
                    for occlusion in ("none", "half_red", "all_red"):
                        image = rgb.copy()
                        _, xx = np.indices(depth.shape)
                        if occlusion == "half_red":
                            image[red & (xx <= np.median(np.nonzero(red)[1]))] = 0
                        elif occlusion == "all_red":
                            image[red] = 0
                        for noise in ("none", "uniform_2mm", "bias_6mm"):
                            changed = depth.copy()
                            if noise == "uniform_2mm":
                                changed += np.random.default_rng(73).uniform(
                                    -0.002, 0.002, depth.shape
                                )
                            elif noise == "bias_6mm":
                                changed += 0.006
                            result = estimate(image, changed, pos, rot, fovy)
                            bound = result.get("bound")
                            contained = (
                                None
                                if bound is None
                                else bool(
                                    np.all(np.array(bound["aabb_min"]) <= truth_low)
                                    and np.all(np.array(bound["aabb_max"]) >= truth_high)
                                )
                            )
                            report["cases"].append(
                                {
                                    "episode": ep,
                                    "camera_shift_x_m": shift,
                                    "occlusion": occlusion,
                                    "depth_perturbation": noise,
                                    "beyond_nominal_error_budget": noise == "bias_6mm",
                                    "estimate": result,
                                    "scoring_only": {
                                        "true_x_gap_m": float(truth_gap),
                                        "full_aabb_contained": contained,
                                        "false_positive_gap": bool(
                                            bound
                                            and bound["conditional_positive_gap"]
                                            and truth_gap <= 0
                                        ),
                                    },
                                }
                            )
            finally:
                if renderer is not None:
                    renderer.close()
                world.close()
        if payload_digest(root) != source:
            raise ValueError("source changed")
        rows = report["cases"]
        report["summary"] = {
            "trials": len(rows),
            "unknown": sum(r["estimate"]["status"] == "unknown" for r in rows),
            "bounds": sum("bound" in r["estimate"] for r in rows),
            "containment_failures": sum(
                r["scoring_only"]["full_aabb_contained"] is False for r in rows
            ),
            "false_positive_contact_gaps": sum(
                r["scoring_only"]["false_positive_gap"] for r in rows
            ),
        }
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    report["limitations"] = [
        "controlled training-fixture stress test, not held-out policy evaluation",
        "camera translation only; orientation unchanged",
        "occlusion is image masking, not a physical occluder",
        "6mm depth bias exceeds assumed 3mm point/wall budgets; passing cannot validate that budget",
        "no swept volume, complete obstacle map or recovery action",
    ]
    Path("docs/evaluations/envelope-stress-v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
