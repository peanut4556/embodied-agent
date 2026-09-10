"""Outcome-labelled LeRobot recordings of the production feedback executor."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .demonstrations import IMAGE_KEY, JOINTS, TASK, features, write_json
from .physics import PhysicsWorld

PHASES = ["running", "checking_grasp", "returning", "completed", "stopped"]
REPO = "local/embodied-agent-recovery"


def recovery_features():
    result = features()
    result["replay.action"] = {"dtype": "float64", "shape": (5,), "names": JOINTS}
    for key in ("observation.holding", "next.success"):
        result[key] = {"dtype": "bool", "shape": (1,), "names": [key]}
    for key in ("controller.retries", "controller.slips"):
        result[key] = {"dtype": "int64", "shape": (1,), "names": [key]}
    result["environment.force"] = {"dtype": "float32", "shape": (3,), "names": ["fx", "fy", "fz"]}
    return result


def collect_episode(model, case):
    world = PhysicsWorld(policy_path=model)
    renderer = None
    rows = []
    try:
        world.data.qpos[5] = case["x"]  # Initial fixture setup only.
        mujoco.mj_forward(world.model, world.data)
        world.tick(0.2)
        initial = {key: getattr(world.data, key).tolist() for key in ("qpos", "qvel", "ctrl")}
        initial["time"] = float(world.data.time)
        initial["qacc_warmstart"] = world.data.qacc_warmstart.tolist()
        world.begin("learned_pick_place", {"object": "red_block", "destination": "box"})
        motion = world.policy_motion
        fps = motion.controller.fps
        body = world.model.body("red_block").id
        start = round(case.get("force_at", -1) * fps)
        end = start + round(case.get("force_duration", 0) * fps)
        renderer = mujoco.Renderer(world.model, height=240, width=320)
        camera = world.model.camera("perception").id
        calibration = {
            "name": "perception",
            "shape": [240, 320, 3],
            "position": world.data.cam_xpos[camera].tolist(),
            "rotation": world.data.cam_xmat[camera].reshape(3, 3).tolist(),
            "fovy": float(world.model.cam_fovy[camera]),
        }

        def capture(current, rgb, holding):
            if abs(current.data.time - initial["time"] - len(rows) / fps) > 1e-7:
                raise ValueError("recovery sampling clock drift")
            ex = motion.controller
            rows.append(
                {
                    "observation.state": current.data.qpos[:5].astype(np.float32).copy(),
                    "observation.velocity": current.data.qvel[:5].astype(np.float32).copy(),
                    "observation.holding": np.array([holding]),
                    IMAGE_KEY: rgb.copy(),
                    "action": current.data.ctrl.astype(np.float32).copy(),
                    "replay.action": current.data.ctrl.copy(),
                    "environment.force": current.data.xfrc_applied[body, :3]
                    .astype(np.float32)
                    .copy(),
                    "phase": np.array([PHASES.index(ex.state)], dtype=np.int64),
                    "controller.retries": np.array([ex.retries], dtype=np.int64),
                    "controller.slips": np.array([ex.slips], dtype=np.int64),
                    "next.done": np.array([False]),
                    "next.success": np.array([False]),
                    "task": TASK,
                }
            )

        stopped_at = None
        for tick in range(32 * fps):
            world.data.xfrc_applied[:] = 0
            if tick == start and not world.holding:
                raise ValueError("force fixture did not begin with a contact grasp")
            if start <= tick < end:
                world.data.xfrc_applied[body, 2] = case["force_z"]
            if case.get("stop_on_recovery") and motion.controller.state == "returning":
                world.begin("stop", {})
            if world.active:
                world.tick(1 / fps, on_control_frame=capture)
            else:
                if stopped_at is None:
                    stopped_at = tick
                mujoco.mj_forward(world.model, world.data)
                renderer.update_scene(world.data, camera="perception")
                capture(world, renderer.render(), world.holding)
                world.tick(1 / fps)
                if tick - stopped_at >= fps - 1:
                    break
        else:
            raise RuntimeError("recovery recording did not terminate")
        ex = motion.snapshot()
        success = ex["state"] == "completed" and world.inside_box()
        rows[-1]["next.done"][:] = True
        rows[-1]["next.success"][:] = success
        episode = {
            "scenario": case,
            "frames": len(rows),
            "initial_state": initial,
            "success": bool(success),
            "inside_box": world.inside_box(),
            "final_xyz": world.snapshot()["block_xyz"],
            "execution": ex,
        }
        return rows, episode, calibration, fps
    finally:
        world.data.xfrc_applied[:] = 0
        if renderer is not None:
            renderer.close()
        world.close()


def record_recovery(root, model, scenarios):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, model = Path(root), Path(model)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {root}")
    if not scenarios or any(not 0.28 <= case["x"] <= 0.40 for case in scenarios):
        raise ValueError("recovery recording requires supported initial positions")
    fps = json.loads((model / "training.json").read_text())["fps"]
    source = Path(__file__).parent / "assets/tabletop.xml"
    manifest = {
        "schema_version": 1,
        "dataset_type": "feedback_recovery",
        "status": "recording",
        "repo_id": REPO,
        "task": TASK,
        "fps": fps,
        "joint_names": JOINTS,
        "phase_names": PHASES,
        "episodes": [],
        "mujoco_version": mujoco.__version__,
        "model_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256((model / "policy.npz").read_bytes()).hexdigest(),
        "policy_metadata_sha256": hashlib.sha256(
            (model / "training.json").read_bytes()
        ).hexdigest(),
        "controller_sha256": hashlib.sha256(
            (source.parent.parent / "feedback.py").read_bytes()
        ).hexdigest(),
        "sampling": "exact policy RGB/contact inputs and new action, before physics integration",
        "action_semantics": "absolute actuator position targets; first 3 rad, last 2 m",
        "training_use": "curation only; previously inspected scenarios, not an independent test set",
        "policy_input_keys": [IMAGE_KEY, "observation.state", "observation.holding"],
        "excluded_input_keys": [
            "replay.action",
            "environment.force",
            "phase",
            "controller.retries",
            "controller.slips",
            "next.done",
            "next.success",
        ],
    }
    dataset = LeRobotDataset.create(
        repo_id=REPO,
        root=root,
        fps=fps,
        features=recovery_features(),
        robot_type="mujoco_tabletop_5dof",
        use_videos=False,
        image_writer_threads=2,
        video_backend="pyav",
    )
    write_json(root / "recording.json", manifest)
    try:
        for index, case in enumerate(scenarios):
            rows, episode, camera, episode_fps = collect_episode(model, case)
            if fps != episode_fps:
                raise ValueError("recording and policy clocks differ")
            for row in rows:
                dataset.add_frame(row)
            dataset.save_episode()
            episode["index"] = index
            manifest["episodes"].append(episode)
            manifest["camera"] = camera
            write_json(root / "recording.json", manifest)
            print(
                f"RECORDED {case['name']}: {len(rows)} frames, success={episode['success']}",
                flush=True,
            )
            del rows
        dataset.finalize()
        manifest["status"] = "recorded"
        write_json(root / "recording.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(root / "recording.json", manifest)
        dataset.finalize()
        raise
    return validate_recovery(root)


def validate_recovery(root):
    root = Path(root)
    manifest = json.loads((root / "recording.json").read_text())
    if manifest.get("dataset_type") != "feedback_recovery" or manifest["status"] not in {
        "recorded",
        "validated",
        "validation_failed",
    }:
        raise ValueError("not a complete recovery recording")
    manifest["status"] = "validating"
    write_json(root / "recording.json", manifest)
    # Invalidate any old training selection before reading or replaying frames.
    write_json(root / "selection.json", {"usable": False})
    try:
        report = replay_recovery(root, manifest)
    except BaseException as exc:
        manifest["status"] = "validation_failed"
        write_json(root / "recording.json", manifest)
        write_json(root / "validation.json", {"success": False, "error": str(exc)})
        raise
    manifest["status"] = "validated"
    write_json(root / "validation.json", report)
    write_json(root / "recording.json", manifest)
    write_json(
        root / "selection.json",
        {
            "usable": True,
            "successful_episodes": [e["index"] for e in manifest["episodes"] if e["success"]],
            "failure_episodes": [e["index"] for e in manifest["episodes"] if not e["success"]],
            "independent_test_episodes": [],
            "note": "Do not use failures as successful imitation targets; this batch is curation data only.",
        },
    )
    return report


def replay_recovery(root, manifest):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = Path(__file__).parent / "assets/tabletop.xml"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("physics model differs from recovery recording")
    if (
        manifest["joint_names"] != JOINTS
        or manifest["phase_names"] != PHASES
        or manifest["fps"] not in {25, 50}
    ):
        raise ValueError("invalid recovery schema")
    dataset = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav")
    if dataset.num_episodes != len(manifest["episodes"]):
        raise ValueError("episode count mismatch")
    # LeRobot's tensor transform casts floating arrays to float32. Read this one
    # diagnostic column from its Arrow backing to retain the declared float64.
    exact_actions = dataset.hf_dataset.data.column("replay.action")
    results, offset = [], 0
    fps = manifest["fps"]
    for episode in manifest["episodes"]:
        world = PhysicsWorld()
        initial = episode["initial_state"]
        for key in ("qpos", "qvel", "ctrl"):
            getattr(world.data, key)[:] = initial[key]
        world.data.time = initial["time"]
        world.data.qacc_warmstart[:] = initial["qacc_warmstart"]
        body = world.model.body("red_block").id
        max_error = np.zeros(5)
        previous_counts = np.zeros(2, dtype=int)
        try:
            for index in range(episode["frames"]):
                row = dataset[offset + index]
                state, action = row["observation.state"].numpy(), row["action"].numpy()
                exact_action = np.asarray(exact_actions[offset + index].as_py(), dtype=np.float64)
                if (
                    exact_action.shape != (5,)
                    or not np.isfinite(exact_action).all()
                    or not np.array_equal(action, exact_action.astype(np.float32))
                ):
                    raise ValueError("training and precise replay actions disagree")
                image = row[IMAGE_KEY].numpy()
                velocity, force = (
                    row["observation.velocity"].numpy(),
                    row["environment.force"].numpy(),
                )
                if (
                    state.shape != (5,)
                    or action.shape != (5,)
                    or velocity.shape != (5,)
                    or force.shape != (3,)
                    or image.shape != (3, 240, 320)
                ):
                    raise ValueError("unexpected recovery frame shape")
                if (
                    not all(np.isfinite(v).all() for v in (state, action, velocity, force, image))
                    or image.min() < 0
                    or image.max() > 1
                    or image.std() < 0.01
                ):
                    raise ValueError("invalid recovery frame values")
                if (
                    abs(row["timestamp"].item() - index / fps) > 1e-4
                    or row["frame_index"].item() != index
                    or row["episode_index"].item() != episode["index"]
                    or row["task"] != TASK
                ):
                    raise ValueError("recovery frame boundary or timestamp mismatch")
                last = index == episode["frames"] - 1
                if bool(row["next.done"].item()) != last or bool(row["next.success"].item()) != (
                    last and episode["success"]
                ):
                    raise ValueError("recovery outcome label mismatch")
                phase = row["phase"].item()
                counts = np.array(
                    [row["controller.retries"].item(), row["controller.slips"].item()]
                )
                if (
                    phase not in range(len(PHASES))
                    or np.any(counts < previous_counts)
                    or np.any(counts - previous_counts > 1)
                ):
                    raise ValueError("invalid recovery controller annotations")
                previous_counts = counts
                expected_force = np.zeros(3)
                case = episode["scenario"]
                start = round(case.get("force_at", -1) * fps)
                if start <= index < start + round(case.get("force_duration", 0) * fps):
                    expected_force[2] = case["force_z"]
                if not np.allclose(force, expected_force, atol=1e-7):
                    raise ValueError("external force differs from recorded fixture")
                if np.any(action < world.model.actuator_ctrlrange[:, 0] - 1e-6) or np.any(
                    action > world.model.actuator_ctrlrange[:, 1] + 1e-6
                ):
                    raise ValueError("action exceeds actuator bounds")
                max_error = np.maximum(max_error, np.abs(world.data.qpos[:5] - state))
                if np.any(max_error > [1e-5, 1e-5, 1e-5, 1e-6, 1e-6]):
                    raise ValueError(
                        f"saved-action replay diverged: episode={episode['index']} frame={index} error={max_error.tolist()}"
                    )
                world.data.xfrc_applied[body, :3] = force
                mujoco.mj_forward(world.model, world.data)
                if not np.allclose(world.data.qvel[:5], velocity, atol=1e-5, rtol=1e-5):
                    raise ValueError("replayed joint velocity differs from observation")
                if bool(row["observation.holding"].item()) != world.holding:
                    raise ValueError("replayed contact differs from policy observation")
                world.data.ctrl[:] = exact_action
                for _ in range(round(1 / fps / world.model.opt.timestep)):
                    mujoco.mj_step(world.model, world.data)
            mujoco.mj_forward(world.model, world.data)
            terminal = episode["execution"]["state"]
            if terminal not in {"completed", "stopped"} or PHASES[phase] != terminal:
                raise ValueError("invalid terminal controller state")
            if counts.tolist() != [episode["execution"]["retries"], episode["execution"]["slips"]]:
                raise ValueError("terminal recovery counters mismatch")
            if bool(episode["success"]) != (terminal == "completed" and world.inside_box()):
                raise ValueError("replayed outcome differs from label")
            if world.inside_box() != episode["inside_box"] or not np.allclose(
                world.snapshot()["block_xyz"], episode["final_xyz"], atol=1e-4
            ):
                raise ValueError("replayed object outcome differs from recording")
            results.append(
                {
                    "episode": episode["index"],
                    "frames": episode["frames"],
                    "replay_matches": True,
                    "task_success": episode["success"],
                    "max_joint_error": max_error.tolist(),
                }
            )
            offset += episode["frames"]
        finally:
            world.close()
    if offset != len(dataset):
        raise ValueError("unexpected extra recovery frames")
    report = {"success": True, "total_frames": offset, "episodes": results}
    print(json.dumps(report), flush=True)
    return report
