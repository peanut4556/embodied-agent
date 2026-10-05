"""Replay exact recovery actions and inspect RGB-D at the retreat-to-pick decision."""

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


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path("outputs/datasets/corrections-contact-stable-v1")
    output = Path("outputs/evaluations/contact-vision-v1")
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "recording.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    source = payload_digest(root)
    if manifest["status"] != "validated" or source != selection["payload_sha256"]:
        raise ValueError("validated unchanged dataset required")
    dataset = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav")
    exact = dataset.hf_dataset.data.column("replay.action")
    offsets = np.cumsum([0] + [e["frames"] for e in manifest["episodes"]])
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "source_sha256": source,
        "script_sha256": digest(__file__),
        "perception_sha256": digest("src/embodied_agent/perception.py"),
        "group": "train",
        "training_performed": False,
        "test_executed": False,
        "cases": [],
    }
    try:
        for ep in (3, 4, 5):
            record = manifest["episodes"][ep]
            tick = record["takeover"]["tick"] + 3 * manifest["fps"] - 1
            world = PhysicsWorld()
            renderer = None
            try:
                for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                    getattr(world.data, key)[:] = record["initial_state"][key]
                world.data.time = record["initial_state"]["time"]
                for t in range(tick + 1):
                    mujoco.mj_forward(world.model, world.data)
                    if not np.allclose(
                        world.data.qpos[:5],
                        dataset[int(offsets[ep]) + t]["observation.state"].numpy(),
                        atol=1e-6,
                        rtol=0,
                    ):
                        raise ValueError("replay state mismatch")
                    if t == tick:
                        break
                    world.data.ctrl[:] = exact[int(offsets[ep]) + t].as_py()
                    for _ in range(round(1 / 25 / world.model.opt.timestep)):
                        mujoco.mj_step(world.model, world.data)
                renderer = mujoco.Renderer(world.model, height=480, width=640)
                renderer.update_scene(world.data, camera="perception")
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                camera = world.model.camera("perception").id
                pos, rot = world.data.cam_xpos[camera].copy(), world.data.cam_xmat[camera].copy()
                diagnostic = {}
                try:
                    detection = locate_red_cube(
                        rgb, depth, pos, rot, world.model.cam_fovy[camera], diagnostic
                    )
                    error = None
                except ValueError as exc:
                    detection, error = None, str(exc)
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                cube = (segmentation[:, :, 0] == world.model.geom("red_block").id) & (
                    segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
                )
                red = (
                    (rgb[:, :, 0] > 70)
                    & (rgb[:, :, 0] > 1.6 * rgb[:, :, 1])
                    & (rgb[:, :, 0] > 1.4 * rgb[:, :, 2])
                )
                prefix = f"episode-{ep}-tick-{tick}"
                Image.fromarray(rgb).save(output / f"{prefix}.png")
                Image.fromarray((cube * 255).astype(np.uint8)).save(
                    output / f"{prefix}-visible-cube.png"
                )
                np.savez_compressed(
                    output / f"{prefix}-rgbd.npz",
                    rgb=rgb,
                    depth=depth,
                    camera_position=pos,
                    camera_rotation=rot,
                    fovy=world.model.cam_fovy[camera],
                )
                orientation = world.data.body("red_block").xmat.reshape(3, 3)
                report["cases"].append(
                    {
                        "episode": ep,
                        "takeover_tick": record["takeover"]["tick"],
                        "decision_tick": tick,
                        "recorded_success": record["success"],
                        "recorded_reason": record["reason"],
                        "detection": detection,
                        "error": error,
                        "diagnostics": diagnostic,
                        "visible_cube_pixels": int(cube.sum()),
                        "red_cube_pixels": int((red & cube).sum()),
                        "truth_diagnostic_only": {
                            "xyz": world.data.body("red_block").xpos.tolist(),
                            "nearest_face_tilt_degrees": float(
                                np.rad2deg(
                                    np.arccos(np.clip(np.max(np.abs(orientation[2, :])), 0, 1))
                                )
                            ),
                        },
                        "image": str(output / f"{prefix}.png"),
                        "rgbd_sha256": digest(output / f"{prefix}-rgbd.npz"),
                    }
                )
                print(json.dumps(report["cases"][-1]), flush=True)
            finally:
                if renderer is not None:
                    renderer.close()
                world.close()
        if payload_digest(root) != source:
            raise ValueError("dataset changed")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/evaluations/contact-vision-v1.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
