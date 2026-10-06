"""Four-second passive hold and RGB-D reinspection of the remaining tilted case."""

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "outputs/hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import mujoco
import numpy as np
from PIL import Image

from embodied_agent.correction_data import payload_digest
from embodied_agent.memory_policy import digest
from embodied_agent.perception import locate_red_cube
from embodied_agent.physics import PhysicsWorld

WAIT_TICKS = (0, 5, 10, 25, 50, 100)


def inspect(world, renderer):
    renderer.update_scene(world.data, camera="perception")
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    cam = world.model.camera("perception").id
    pos, rot = world.data.cam_xpos[cam].copy(), world.data.cam_xmat[cam].copy()
    diagnostics = {}
    try:
        detection = locate_red_cube(
            rgb, depth, pos, rot, world.model.cam_fovy[cam], diagnostics, size_mode="oriented"
        )
        error = None
    except ValueError as exc:
        detection, error = None, str(exc)
    return (
        rgb,
        depth,
        pos,
        rot,
        {
            "detection": detection,
            "error": error,
            "diagnostics": diagnostics,
            "grasp_geometry_eligible": bool(
                detection is not None
                and not world.holding
                and abs(detection["xyz"][1]) <= 0.015
                and detection["xyz"][2] >= 0.02
            ),
        },
    )


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path("outputs/datasets/corrections-contact-oriented-v1")
    output = Path("outputs/evaluations/bounded-wait-v1")
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "recording.json").read_text())
    source = payload_digest(root)
    if (
        manifest["status"] != "validated"
        or json.loads((root / "selection.json").read_text())["payload_sha256"] != source
    ):
        raise ValueError("validated unchanged dataset required")
    episode = manifest["episodes"][5]
    if (
        episode["takeover"]["tick"] != 84
        or episode["scenario"]["x"] != 0.302
        or episode["success"]
        or 5 not in manifest["split"]["train"]
    ):
        raise ValueError("unexpected remaining failure fixture")
    dataset = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav")
    offset = sum(e["frames"] for e in manifest["episodes"][:5])
    exact = dataset.hf_dataset.data.column("replay.action")
    world = PhysicsWorld(perception="rgbd", perception_size_mode="oriented")
    renderer = None
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "source_sha256": source,
        "script_sha256": digest(__file__),
        "episode": 5,
        "takeover_tick": 84,
        "decision_tick": 158,
        "wait_ticks": WAIT_TICKS,
        "fps": 25,
        "group": "train",
        "training_performed": False,
        "test_executed": False,
        "observations": [],
    }
    try:
        for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
            getattr(world.data, key)[:] = episode["initial_state"][key]
        world.data.time = episode["initial_state"]["time"]
        for tick in range(159):
            mujoco.mj_forward(world.model, world.data)
            if not np.allclose(
                world.data.qpos[:5],
                dataset[offset + tick]["observation.state"].numpy(),
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError("replay joint mismatch")
            if tick == 158:
                break
            world.data.ctrl[:] = exact[offset + tick].as_py()
            for _ in range(round(0.04 / world.model.opt.timestep)):
                mujoco.mj_step(world.model, world.data)
        command = world.data.ctrl.copy()
        report["held_command"] = command.tolist()
        renderer = mujoco.Renderer(world.model, height=480, width=640)
        initial_xyz = world.data.body("red_block").xpos.copy()
        for tick in range(WAIT_TICKS[-1] + 1):
            mujoco.mj_forward(world.model, world.data)
            if not np.array_equal(world.data.ctrl, command):
                raise ValueError("wait modified control")
            if tick in WAIT_TICKS:
                rgb, depth, pos, rot, row = inspect(world, renderer)
                prefix = f"wait-{tick:03d}"
                Image.fromarray(rgb).save(output / f"{prefix}.png")
                np.savez_compressed(
                    output / f"{prefix}.npz",
                    rgb=rgb,
                    depth=depth,
                    camera_position=pos,
                    camera_rotation=rot,
                    fovy=world.model.cam_fovy[world.model.camera("perception").id],
                )
                orientation = world.data.body("red_block").xmat.reshape(3, 3)
                row.update(
                    wait_seconds=tick / 25,
                    simulation_time=float(world.data.time),
                    image=str(output / f"{prefix}.png"),
                    rgbd_sha256=digest(output / f"{prefix}.npz"),
                    truth_diagnostic_only={
                        "xyz": world.data.body("red_block").xpos.tolist(),
                        "displacement_m": float(
                            np.linalg.norm(world.data.body("red_block").xpos - initial_xyz)
                        ),
                        "speed_m_s": float(np.linalg.norm(world.data.qvel[5:8])),
                        "nearest_face_tilt_degrees": float(
                            np.rad2deg(np.arccos(np.clip(np.max(np.abs(orientation[2, :])), 0, 1)))
                        ),
                    },
                )
                report["observations"].append(row)
                print(json.dumps(row), flush=True)
            if tick < WAIT_TICKS[-1]:
                for _ in range(round(0.04 / world.model.opt.timestep)):
                    mujoco.mj_step(world.model, world.data)
        if payload_digest(root) != source:
            raise ValueError("dataset changed")
        report["status"] = "complete"
        report["eligible_observations"] = sum(
            r["grasp_geometry_eligible"] for r in report["observations"]
        )
        report["limitations"] = [
            "one fixed training failure; only sampled reinspection times",
            "held existing actuator commands; no reposition or physical recovery action",
            "truth diagnostics never used by perception",
            "eligibility is not successful grasping; no retraining or deployment",
        ]
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        if renderer is not None:
            renderer.close()
        world.close()
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/evaluations/bounded-wait-v1.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
