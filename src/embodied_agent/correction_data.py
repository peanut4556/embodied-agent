"""Actual learner roll-in followed by RGB-D expert correction, with exact replay."""

from pathlib import Path

import mujoco
import numpy as np

from .demonstrations import IMAGE_KEY, JOINTS, TASK, write_json
from .grasp_quality import GraspQuality
from .memory_policy import MemoryPolicy, digest
from .physics import PhysicsWorld
from .reactive import ReactiveExecutor
from .recovery_data import recovery_features
from .temporal_data import read_json

SOURCES = ["learner", "expert", "settle", "stopped"]
REPO = "local/embodied-agent-corrections"


def check_layout(manifest):
    episodes, split = manifest["episodes"], manifest["split"]
    if (
        manifest["fps"] not in (25, 50)
        or manifest["joint_names"] != JOINTS
        or manifest["source_names"] != SOURCES
        or not episodes
    ):
        raise ValueError("invalid correction schema")
    if set(split) != {"train", "validation", "test"} or split["test"]:
        raise ValueError("correction split must be development-only")
    ids = [i for values in split.values() for i in values]
    if any(type(i) is not int for i in ids) or sorted(ids) != list(range(len(episodes))):
        raise ValueError("correction split must cover episodes exactly once")
    scenes = {}
    for group, indices in split.items():
        for index in indices:
            episode = episodes[index]
            case = episode["scenario"]
            x, seconds = case["x"], case["takeover_seconds"]
            if (
                not np.isfinite([x, seconds]).all()
                or not 0.28 <= x <= 0.40
                or not 0 < seconds < 10
                or episode["index"] != index
                or type(episode["success"]) is not bool
                or type(episode["frames"]) is not int
                or not 0 < episode["takeover"]["tick"] < episode["frames"]
                or episode["takeover"]["tick"] != round(seconds * manifest["fps"])
            ):
                raise ValueError("invalid correction episode or takeover")
            x = round(x, 9)
            if x in scenes and scenes[x] != group:
                raise ValueError("same correction position crosses groups")
            scenes[x] = group


def correction_features():
    features = recovery_features()
    for key in ("controller.retries", "controller.slips", "environment.force", "phase"):
        del features[key]
    features["controller.source"] = {"dtype": "int64", "shape": (1,), "names": ["source"]}
    features["supervision.valid"] = {"dtype": "bool", "shape": (1,), "names": ["valid"]}
    return features


class CorrectionExpert:
    """Finite RGB-D/IK supervisor; all targets execute at the recording clock."""

    def __init__(self, world, fps):
        self.world, self.fps = world, fps
        self.stage, self.queue, self.index = "retreat", [], 0
        self.start = world.data.qpos[:5].copy()
        self.command = world.data.ctrl.copy()
        self.origin = None
        self.reason, self.done = "", False
        self.detection = None
        # Move to the established observation posture from the actual arm state.
        # Preserve a contact grasp while retreating; do not teleport the object.
        self.retreat_target = np.r_[
            world.ik(0.22, 0.30), [0.0, 0.0] if world.holding else [0.04, 0.04]
        ]
        self.retreat_grasp = world.holding
        self.rate = np.array([2.0, 2.0, 2.0, 1.0, 1.0]) / fps

    def stop(self, reason):
        if not self.reason:
            self.reason, self.done = reason, True
            self.command[:3] = np.clip(
                self.world.data.qpos[:3],
                self.world.model.actuator_ctrlrange[:3, 0],
                self.world.model.actuator_ctrlrange[:3, 1],
            )
        return self.command.copy()

    def _begin(self, action):
        self.world.begin(action, {"object": "red_block", "destination": "box"})
        self.queue = list(self.world.motion)
        self.world.motion = []
        self.world.active = ""
        self.detection = self.world.detection
        self.stage, self.index = action, 0
        self.origin = self.world.data.site("tip").xpos[[0, 2]].copy()

    def step(self):
        if self.done:
            return self.command.copy()
        try:
            if self.stage == "retreat":
                self.index += 1
                t = min(1.0, self.index / (2 * self.fps))
                target = self.start + (self.retreat_target - self.start) * (t * t * (3 - 2 * t))
                if self.index >= 3 * self.fps:
                    if not np.allclose(self.world.data.qpos[:5], self.retreat_target, atol=0.015):
                        return self.stop("expert retreat did not settle")
                    self._begin("place" if self.world.holding else "pick")
            else:
                if not self.queue:
                    if self.stage == "pick":
                        if not self.world.holding:
                            return self.stop("expert failed to establish grasp")
                        self._begin("place")
                    else:
                        self.done = True
                        return self.command.copy()
                point, grip, duration = self.queue[0]
                self.index += 1
                t = min(1.0, self.index / round(duration * self.fps))
                tip = self.origin + (np.asarray(point) - self.origin) * (t * t * (3 - 2 * t))
                target = np.r_[self.world.ik(*tip), grip, grip]
                if t >= 1:
                    self.queue.pop(0)
                    self.origin, self.index = np.asarray(point), 0
            target = np.clip(
                target,
                self.world.model.actuator_ctrlrange[:, 0],
                self.world.model.actuator_ctrlrange[:, 1],
            )
            self.command += np.clip(target - self.command, -self.rate, self.rate)
            return self.command.copy()
        except (ValueError, RuntimeError) as exc:
            return self.stop(f"expert rejected state: {exc}")


def collect(policy, case):
    world = PhysicsWorld(perception="rgbd")
    renderer = None
    try:
        world.data.qpos[5] = case["x"]
        mujoco.mj_forward(world.model, world.data)
        world.tick(0.2)
        initial = {
            key: getattr(world.data, key).tolist()
            for key in ("qpos", "qvel", "ctrl", "qacc_warmstart")
        }
        initial["time"] = float(world.data.time)
        fps = policy.metadata["fps"]
        learner = ReactiveExecutor(policy.session(), world.data.qpos[:5], max_seconds=30)
        renderer = mujoco.Renderer(world.model, height=240, width=320)
        quality = GraspQuality(fps)
        rows, expert, takeover, terminal = [], None, None, None
        takeover_tick = round(case["takeover_seconds"] * fps)
        for tick in range(30 * fps):
            mujoco.mj_forward(world.model, world.data)
            renderer.update_scene(world.data, camera="perception")
            rgb, joints, holding = renderer.render(), world.data.qpos[:5].copy(), world.holding
            if tick == takeover_tick:
                takeover = {
                    key: getattr(world.data, key).tolist() for key in ("qpos", "qvel", "ctrl")
                }
                takeover["tick"] = tick
                expert = CorrectionExpert(world, fps)
            if "stop_seconds" in case and tick >= round(case["stop_seconds"] * fps):
                command = expert.stop("user stop") if expert else learner.stop(joints)
                source = 3
            elif expert:
                command = expert.step()
                source = 3 if expert.reason else (2 if expert.done else 1)
            else:
                command = learner.step(rgb, joints, holding)
                source = 0
                if learner.state == "stopped":
                    raise RuntimeError("learner stopped before scheduled expert takeover")
            quality.observe(
                holding, world.data.body("red_block").xpos.copy(), command, world.inside_box()
            )
            rows.append(
                {
                    IMAGE_KEY: rgb.copy(),
                    "observation.state": joints.astype(np.float32),
                    "observation.velocity": world.data.qvel[:5].astype(np.float32).copy(),
                    "observation.holding": np.array([holding]),
                    "action": command.astype(np.float32),
                    "replay.action": command.copy(),
                    "controller.source": np.array([source], dtype=np.int64),
                    "supervision.valid": np.array([False]),
                    "next.done": np.array([False]),
                    "next.success": np.array([False]),
                    "task": TASK,
                }
            )
            world.data.ctrl[:] = command
            for _ in range(round(1 / fps / world.model.opt.timestep)):
                mujoco.mj_step(world.model, world.data)
            if source in (2, 3):
                if terminal is None:
                    terminal = tick
                if tick - terminal >= fps:
                    break
        else:
            raise RuntimeError("correction collection timeout")
        mujoco.mj_forward(world.model, world.data)
        result = quality.result()
        success = bool(
            result["verified_pick_place"] and world.inside_box() and expert and not expert.reason
        )
        rows[-1]["next.done"][:] = True
        rows[-1]["next.success"][:] = success
        for row in rows:
            row["supervision.valid"][:] = success and row["controller.source"].item() == 1
        episode = {
            "scenario": case,
            "frames": len(rows),
            "initial_state": initial,
            "takeover": takeover,
            "success": success,
            "quality": result,
            "reason": expert.reason if expert else "no takeover",
            "detection": expert.detection if expert else None,
            "final_xyz": world.snapshot()["block_xyz"],
            "supervised_frames": sum(r["supervision.valid"].item() for r in rows),
        }
        return rows, episode
    finally:
        if renderer is not None:
            renderer.close()
        world.close()


def record(root, model, scenarios):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite correction data: {root}")
    config = read_json(scenarios)
    cases, split = config["cases"], config["split"]
    ids = [i for group in ("train", "validation", "test") for i in split[group]]
    if sorted(ids) != list(range(len(cases))) or split["test"]:
        raise ValueError("correction data must be development-only and split exactly once")
    groups = {}
    for group, indices in split.items():
        for i in indices:
            x = cases[i]["x"]
            if (
                not np.isfinite(x)
                or not 0.28 <= x <= 0.40
                or not 0 < cases[i]["takeover_seconds"] < 10
            ):
                raise ValueError("invalid correction fixture")
            if x in groups and groups[x] != group:
                raise ValueError("same position crosses correction splits")
            groups[x] = group
    policy = MemoryPolicy(model)
    scene = Path(__file__).parent / "assets/tabletop.xml"
    if digest(scene) != policy.metadata["model_sha256"]:
        raise ValueError("learner checkpoint physics mismatch")
    dataset = LeRobotDataset.create(
        repo_id=REPO,
        root=root,
        fps=policy.metadata["fps"],
        features=correction_features(),
        robot_type="mujoco_tabletop_5dof",
        use_videos=False,
        image_writer_threads=2,
        video_backend="pyav",
    )
    manifest = {
        "schema_version": 1,
        "dataset_type": "feedback_correction",
        "status": "recording",
        "repo_id": REPO,
        "fps": policy.metadata["fps"],
        "joint_names": JOINTS,
        "source_names": SOURCES,
        "episodes": [],
        "split": split,
        "model_sha256": digest(scene),
        "learner_weights_sha256": policy.metadata["weights_sha256"],
        "scenarios_sha256": digest(scenarios),
        "collector_sha256": digest(__file__),
        "policy_input_keys": [IMAGE_KEY, "observation.state", "observation.holding"],
        "excluded_input_keys": [
            "controller.source",
            "supervision.valid",
            "replay.action",
            "next.done",
            "next.success",
        ],
        "sampling": "actual pre-action RGB/state/contact; executed action held for entire frame",
        "training_use": "successful expert frames only; learner prefix is context, never imitation target",
    }
    write_json(root / "recording.json", manifest)
    try:
        for i, case in enumerate(cases):
            rows, episode = collect(policy, case)
            for row in rows:
                dataset.add_frame(row)
            dataset.save_episode()
            episode["index"] = i
            manifest["episodes"].append(episode)
            write_json(root / "recording.json", manifest)
            print(
                f"RECORDED {case['name']}: success={episode['success']}, reason={episode['reason']}, frames={len(rows)}",
                flush=True,
            )
        dataset.finalize()
        manifest["status"] = "recorded"
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        dataset.finalize()
        raise
    finally:
        write_json(root / "recording.json", manifest)
    return validate(root)


def validate(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    manifest = read_json(root / "recording.json")
    if manifest.get("dataset_type") != "feedback_correction" or manifest["status"] not in (
        "recorded",
        "validated",
        "validation_failed",
    ):
        raise ValueError("not a complete correction recording")
    write_json(root / "selection.json", {"usable": False})
    manifest["status"] = "validating"
    write_json(root / "recording.json", manifest)
    try:
        check_layout(manifest)
        if digest(Path(__file__).parent / "assets/tabletop.xml") != manifest["model_sha256"]:
            raise ValueError("correction physics mismatch")
        dataset = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav")
        exact = dataset.hf_dataset.data.column("replay.action")
        offset, results = 0, []
        for episode in manifest["episodes"]:
            world = PhysicsWorld()
            quality = GraspQuality(manifest["fps"])
            try:
                initial = episode["initial_state"]
                for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                    getattr(world.data, key)[:] = initial[key]
                world.data.time = initial["time"]
                maximum, supervised, previous_source = np.zeros(5), 0, 0
                stop_reference = None
                for t in range(episode["frames"]):
                    row = dataset[offset + t]
                    command = np.asarray(exact[offset + t].as_py(), dtype=float)
                    if (
                        command.shape != (5,)
                        or not np.isfinite(command).all()
                        or not np.array_equal(command.astype(np.float32), row["action"].numpy())
                    ):
                        raise ValueError("precise and training action disagree")
                    if np.any(command < world.model.actuator_ctrlrange[:, 0] - 1e-6) or np.any(
                        command > world.model.actuator_ctrlrange[:, 1] + 1e-6
                    ):
                        raise ValueError("correction action exceeds bounds")
                    if (
                        row["episode_index"].item() != episode["index"]
                        or row["frame_index"].item() != t
                        or abs(row["timestamp"].item() - t / manifest["fps"]) > 1e-4
                    ):
                        raise ValueError("correction frame boundary mismatch")
                    maximum = np.maximum(
                        maximum, np.abs(world.data.qpos[:5] - row["observation.state"].numpy())
                    )
                    if np.any(maximum > [1e-5] * 3 + [1e-6] * 2):
                        raise ValueError("correction action replay diverged")
                    mujoco.mj_forward(world.model, world.data)
                    if world.holding != bool(row["observation.holding"].item()) or not np.allclose(
                        world.data.qvel[:5],
                        row["observation.velocity"].numpy(),
                        atol=1e-5,
                        rtol=1e-5,
                    ):
                        raise ValueError("correction observation differs on replay")
                    rgb = row[IMAGE_KEY].numpy()
                    if (
                        rgb.shape != (3, 240, 320)
                        or not np.isfinite(rgb).all()
                        or rgb.min() < 0
                        or rgb.max() > 1
                        or rgb.std() < 0.01
                    ):
                        raise ValueError("invalid correction RGB")
                    source = row["controller.source"].item()
                    if source < previous_source or row["task"] != TASK:
                        raise ValueError("invalid correction controller sequence or task")
                    previous_source = source
                    if source == 3:
                        if stop_reference is None:
                            stop_reference = command.copy()
                        elif not np.array_equal(stop_reference, command):
                            raise ValueError("correction stop command was not latched")
                    if source not in range(4) or (t < episode["takeover"]["tick"]) != (source == 0):
                        raise ValueError("invalid takeover source boundary")
                    valid = bool(row["supervision.valid"].item())
                    if valid != (episode["success"] and source == 1):
                        raise ValueError("unsafe correction supervision mask")
                    supervised += valid
                    last = t == episode["frames"] - 1
                    if bool(row["next.done"].item()) != last or bool(
                        row["next.success"].item()
                    ) != (last and episode["success"]):
                        raise ValueError("correction terminal label mismatch")
                    if t == episode["takeover"]["tick"]:
                        for key in ("qpos", "qvel", "ctrl"):
                            if not np.allclose(
                                getattr(world.data, key),
                                episode["takeover"][key],
                                atol=1e-7,
                                rtol=1e-7,
                            ):
                                raise ValueError("takeover state was not preserved")
                    quality.observe(
                        world.holding,
                        world.data.body("red_block").xpos.copy(),
                        command,
                        world.inside_box(),
                    )
                    world.data.ctrl[:] = command
                    for _ in range(round(1 / manifest["fps"] / world.model.opt.timestep)):
                        mujoco.mj_step(world.model, world.data)
                mujoco.mj_forward(world.model, world.data)
                observed = quality.result()
                success = (
                    observed["verified_pick_place"] and world.inside_box() and not episode["reason"]
                )
                if (
                    observed != episode["quality"]
                    or success != episode["success"]
                    or supervised != episode["supervised_frames"]
                    or not np.allclose(
                        world.snapshot()["block_xyz"], episode["final_xyz"], atol=1e-4
                    )
                ):
                    raise ValueError("correction quality or outcome replay mismatch")
                results.append(
                    {
                        "episode": episode["index"],
                        "replay_matches": True,
                        "success": success,
                        "supervised_frames": supervised,
                        "max_joint_error": maximum.tolist(),
                    }
                )
                offset += episode["frames"]
            finally:
                world.close()
        if offset != len(dataset) or len(results) != dataset.num_episodes:
            raise ValueError("correction dataset count mismatch")
        report = {"success": True, "total_frames": offset, "episodes": results}
        write_json(root / "validation.json", report)
        manifest["status"] = "validated"
        write_json(root / "recording.json", manifest)
        write_json(
            root / "selection.json",
            {
                "usable": True,
                "split": manifest["split"],
                "successful_episodes": [e["index"] for e in manifest["episodes"] if e["success"]],
                "failure_episodes": [e["index"] for e in manifest["episodes"] if not e["success"]],
                "independent_test_episodes": [],
                "required_loss_mask": "supervision.valid",
                "payload_sha256": payload_digest(root),
            },
        )
        return report
    except BaseException as exc:
        manifest["status"] = "validation_failed"
        write_json(root / "validation.json", {"success": False, "error": str(exc)})
        raise
    finally:
        write_json(root / "recording.json", manifest)


def payload_digest(root):
    import hashlib

    root = Path(root)
    hash_state = hashlib.sha256()
    paths = [root / "recording.json"]
    for directory in ("data", "images", "meta"):
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file())
    for path in sorted(paths):
        hash_state.update(str(path.relative_to(root)).encode() + b"\0")
        hash_state.update(bytes.fromhex(digest(path)))
    return hash_state.hexdigest()


def training_sequences(root, group):
    """Load successful correction episodes with a mandatory expert-only loss mask.

    Learner history remains causal context. Its executed actions are never returned
    as imitation targets. This reader does not fit or update any model.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if group not in ("train", "validation"):
        raise ValueError("correction data is development-only")
    manifest, selection = read_json(root / "recording.json"), read_json(root / "selection.json")
    if (
        manifest.get("dataset_type") != "feedback_correction"
        or manifest["status"] != "validated"
        or selection.get("usable") is not True
        or read_json(root / "validation.json").get("success") is not True
        or selection.get("payload_sha256") != payload_digest(root)
    ):
        raise ValueError("validated unchanged correction data required")
    check_layout(manifest)
    dataset = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav")
    offsets = np.cumsum([0] + [e["frames"] for e in manifest["episodes"]])
    for ep in manifest["split"][group]:
        episode = manifest["episodes"][ep]
        if not episode["success"]:
            continue
        inputs = {key: [] for key in manifest["policy_input_keys"]}
        # Strict whitelist, even if someone changes the manifest before revalidation.
        if set(inputs) != {IMAGE_KEY, "observation.state", "observation.holding"}:
            raise ValueError("correction input whitelist mismatch")
        targets, masks = [], []
        for t in range(episode["frames"]):
            row = dataset[int(offsets[ep]) + t]
            valid = bool(row["supervision.valid"].item())
            if valid != (row["controller.source"].item() == 1):
                raise ValueError("correction loss mask mismatch")
            for key, observations in inputs.items():
                value = row[key].numpy()
                observations.append(np.atleast_1d(value) if key == "observation.holding" else value)
            targets.append(row["action"].numpy() if valid else np.zeros(5, dtype=np.float32))
            masks.append(valid)
        yield {
            "episode": ep,
            "inputs": {k: np.stack(v) for k, v in inputs.items()},
            "targets": np.stack(targets),
            "loss_mask": np.array(masks, dtype=bool),
        }
