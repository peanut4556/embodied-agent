"""Local LeRobot demonstration collection from the RGB-D scripted expert.

No ROS service or network is used: each recording owns an isolated physics world.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .physics import PhysicsWorld

JOINTS = ["shoulder", "elbow", "wrist", "finger_left", "finger_right"]
TASK = "把桌上的红色积木放进盒子"
IMAGE_KEY = "observation.images.overhead"
IMAGE_SHAPE = (240, 320, 3)
REPO_ID = "local/embodied-agent-demonstrations"


def features():
    return {
        "observation.state": {"dtype": "float32", "shape": (5,), "names": JOINTS},
        "observation.velocity": {"dtype": "float32", "shape": (5,), "names": JOINTS},
        "action": {"dtype": "float32", "shape": (5,), "names": JOINTS},
        IMAGE_KEY: {
            "dtype": "image",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "phase": {"dtype": "int64", "shape": (1,), "names": ["phase"]},
        "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
    }


def validate_options(positions, fps):
    if fps not in {25, 50}:
        raise ValueError("fps must be 25 or 50, dividing the 500 Hz physics clock")
    if not positions or not all(np.isfinite(x) and 0.28 <= x <= 0.40 for x in positions):
        raise ValueError("cube x positions must be finite and within [0.28, 0.40] m")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def record_dataset(root: Path, positions=(0.28, 0.32, 0.40), fps=25):
    """Record successful episodes; incomplete runs remain explicitly marked on disk."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root).resolve()
    validate_options(positions, fps)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {root}")
    source = Path(__file__).parent / "assets/tabletop.xml"
    manifest = {
        "schema_version": 1,
        "status": "recording",
        "repo_id": REPO_ID,
        "task": TASK,
        "fps": fps,
        "physics_timestep": 0.002,
        "mujoco_version": mujoco.__version__,
        "model_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "joint_names": JOINTS,
        "joint_units": ["rad", "rad", "rad", "m", "m"],
        "action_semantics": "absolute actuator position targets at observation time",
        "sampling": "pre-integration at fixed fps; scripted expert updates at 500 Hz",
        "expert": "RGB-D localization + scripted IK trajectory; no learned policy",
        "phase_names": ["pick", "place", "settle"],
        "episodes": [],
    }
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=root,
        fps=fps,
        features=features(),
        robot_type="mujoco_tabletop_5dof",
        use_videos=False,
        image_writer_threads=2,
        video_backend="pyav",
    )
    write_json(root / "recording.json", manifest)
    try:
        for episode_index, x in enumerate(positions):
            world = PhysicsWorld(perception="rgbd")
            renderer = None
            try:
                # Only initial scene setup changes the free object's pose.
                world.data.qpos[5] = x
                mujoco.mj_forward(world.model, world.data)
                world.tick(0.2)
                initial = {
                    "qpos": world.data.qpos.tolist(),
                    "qvel": world.data.qvel.tolist(),
                    "ctrl": world.data.ctrl.tolist(),
                    "time": float(world.data.time),
                }
                renderer = mujoco.Renderer(world.model, height=240, width=320)
                camera = world.model.camera("perception").id
                if not manifest.get("camera"):
                    manifest["camera"] = {
                        "name": "perception",
                        "shape": list(IMAGE_SHAPE),
                        "position": world.data.cam_xpos[camera].tolist(),
                        "rotation": world.data.cam_xmat[camera].reshape(3, 3).tolist(),
                        "fovy": float(world.model.cam_fovy[camera]),
                    }
                progress = {"count": 0, "phase": 0, "done": False}

                def capture(current, start=initial["time"], camera_renderer=renderer, p=progress):
                    timestamp = current.data.time - start
                    if abs(timestamp - p["count"] / fps) > 1e-7:
                        raise ValueError("camera/state sampling clock drifted")
                    # Refresh derived body poses so image and qpos describe the same instant.
                    mujoco.mj_forward(current.model, current.data)
                    camera_renderer.update_scene(current.data, camera="perception")
                    dataset.add_frame(
                        {
                            "observation.state": current.data.qpos[:5].astype(np.float32).copy(),
                            "observation.velocity": current.data.qvel[:5].astype(np.float32).copy(),
                            "action": current.data.ctrl.astype(np.float32).copy(),
                            IMAGE_KEY: camera_renderer.render(),
                            "phase": np.array([p["phase"]], dtype=np.int64),
                            "next.done": np.array([p["done"]]),
                            "task": TASK,
                        }
                    )
                    p["count"] += 1

                for phase, action in enumerate(("pick", "place")):
                    progress["phase"] = phase
                    world.begin(action, {"object": "red_block", "destination": "box"})
                    for _ in range(12 * fps):
                        finished = world.tick(1 / fps, on_control_frame=capture)
                        if world.error:
                            raise RuntimeError(world.error)
                        if finished:
                            break
                    else:
                        raise RuntimeError(f"expert {action} trajectory timeout")
                progress["phase"] = 2
                for settle_index in range(fps):
                    progress["done"] = settle_index == fps - 1
                    world.tick(1 / fps, on_control_frame=capture)
                if not world.inside_box() or world.holding:
                    raise RuntimeError("recording failed physical goal verification")
                dataset.save_episode()
                manifest["episodes"].append(
                    {
                        "index": episode_index,
                        "initial_cube_x": x,
                        "frames": progress["count"],
                        "initial_state": initial,
                        "success": True,
                        "detection": world.detection,
                        "final_state": world.snapshot(),
                    }
                )
                write_json(root / "recording.json", manifest)
                print(
                    f"PASS episode {episode_index}: x={x:.3f}, {progress['count']} frames",
                    flush=True,
                )
            finally:
                if renderer is not None:
                    renderer.close()
                world.close()
        dataset.finalize()
        manifest["status"] = "recorded"
        write_json(root / "recording.json", manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_json(root / "recording.json", manifest)
        dataset.finalize()
        raise
    return validate_dataset(root)


def _validate_dataset(root: Path, manifest):
    """Read every frame using LeRobot and replay its saved 25/50 Hz actuator targets."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image, ImageDraw

    source = Path(__file__).parent / "assets/tabletop.xml"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("physics model differs from recorded model")
    fps = manifest["fps"]
    validate_options([episode["initial_cube_x"] for episode in manifest["episodes"]], fps)
    if manifest["joint_names"] != JOINTS:
        raise ValueError("joint ordering mismatch")
    dataset = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav")
    if dataset.num_episodes != len(manifest["episodes"]):
        raise ValueError("episode count mismatch")
    offset = 0
    results = []
    preview = []
    for episode in manifest["episodes"]:
        world = PhysicsWorld()
        initial = episode["initial_state"]
        world.data.qpos[:] = initial["qpos"]
        world.data.qvel[:] = initial["qvel"]
        world.data.ctrl[:] = initial["ctrl"]
        world.data.time = initial["time"]
        mujoco.mj_forward(world.model, world.data)
        max_arm_error = 0.0
        max_finger_error = 0.0
        phases = []
        try:
            for index in range(episode["frames"]):
                row = dataset[offset + index]
                state = row["observation.state"].numpy()
                action = row["action"].numpy()
                image = row[IMAGE_KEY].numpy()
                velocity = row["observation.velocity"].numpy()
                if (
                    state.shape != (5,)
                    or action.shape != (5,)
                    or velocity.shape != (5,)
                    or image.shape != (3, 240, 320)
                ):
                    raise ValueError("unexpected frame dimensions")
                if not all(np.isfinite(value).all() for value in (state, action, image, velocity)):
                    raise ValueError("non-finite sample")
                if image.min() < 0 or image.max() > 1 or image.std() < 0.01:
                    raise ValueError("invalid or blank image")
                if episode["index"] == 0 and index % 5 == 0:
                    frame = Image.fromarray(
                        np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    ImageDraw.Draw(frame).text(
                        (6, 6),
                        f"Episode 0 | t={index / fps:.2f}s",
                        fill="white",
                        stroke_width=1,
                        stroke_fill="black",
                    )
                    preview.append(frame)
                if abs(row["timestamp"].item() - index / fps) > 1e-4:
                    raise ValueError("timestamp mismatch")
                if row["episode_index"].item() != episode["index"]:
                    raise ValueError("episode boundary mismatch")
                if row["frame_index"].item() != index or row["task"] != TASK:
                    raise ValueError("frame index or task mismatch")
                if bool(row["next.done"].item()) != (index == episode["frames"] - 1):
                    raise ValueError("terminal frame mismatch")
                phases.append(row["phase"].item())
                if np.any(action < world.model.actuator_ctrlrange[:, 0] - 1e-6) or np.any(
                    action > world.model.actuator_ctrlrange[:, 1] + 1e-6
                ):
                    raise ValueError("action exceeds actuator bounds")
                error = np.abs(world.data.qpos[:5] - state)
                max_arm_error = max(max_arm_error, float(np.max(error[:3])))
                max_finger_error = max(max_finger_error, float(np.max(error[3:])))
                # Replay only stored actions; no IK, RGB-D detection or expert is called.
                world.data.ctrl[:] = action
                for _ in range(round(1 / fps / world.model.opt.timestep)):
                    mujoco.mj_step(world.model, world.data)
            if not world.inside_box() or world.holding:
                raise ValueError(f"saved-action replay failed for episode {episode['index']}")
            if sorted(set(phases)) != [0, 1, 2] or phases != sorted(phases):
                raise ValueError("invalid action phase sequence")
            results.append(
                {
                    "episode": episode["index"],
                    "frames": episode["frames"],
                    "replay_success": True,
                    "max_arm_error_rad": max_arm_error,
                    "max_finger_error_m": max_finger_error,
                }
            )
            offset += episode["frames"]
        finally:
            world.close()
    if offset != len(dataset):
        raise ValueError("unexpected extra dataset frames")
    report = {"episodes": results, "total_frames": offset, "fps": fps, "success": True}
    preview[0].save(
        root / "preview.gif",
        save_all=True,
        append_images=preview[1:],
        duration=round(5000 / fps),
        loop=0,
    )
    write_json(root / "validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def validate_dataset(root: Path):
    """Never leave a previous success report valid after revalidation fails."""
    root = Path(root).resolve()
    manifest = json.loads((root / "recording.json").read_text())
    if manifest["status"] not in {"recorded", "validated", "validation_failed"}:
        raise ValueError("dataset is incomplete or failed")
    manifest["status"] = "validating"
    write_json(root / "recording.json", manifest)
    try:
        report = _validate_dataset(root, manifest)
    except BaseException as exc:
        write_json(root / "validation.json", {"success": False, "error": str(exc)})
        manifest["status"] = "validation_failed"
        write_json(root / "recording.json", manifest)
        raise
    manifest["status"] = "validated"
    write_json(root / "recording.json", manifest)
    return report
