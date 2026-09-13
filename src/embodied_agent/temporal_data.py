"""Causal, episode-local windows for future feedback policy training."""

import hashlib
import json
from pathlib import Path

import numpy as np

IMAGE_KEY = "observation.images.overhead"
INPUT_KEYS = (IMAGE_KEY, "observation.state", "observation.holding")
GROUPS = ("train", "validation", "test")


def read_json(path):
    return json.loads(Path(path).read_text())


def source_digest(root):
    """Bind the index to metadata, annotations, images and stored frame values."""
    digest = hashlib.sha256()
    paths = [root / name for name in ("recording.json", "selection.json", "validation.json")]
    for directory in ("data", "meta", "images"):
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file())
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def check_source(root, split):
    manifest = read_json(root / "recording.json")
    selection = read_json(root / "selection.json")
    validation = read_json(root / "validation.json")
    if (
        manifest.get("dataset_type") != "feedback_recovery"
        or manifest.get("status") != "validated"
        or selection.get("usable") is not True
        or validation.get("success") is not True
    ):
        raise ValueError("requires validated, usable recovery data")
    episodes = manifest["episodes"]
    if manifest["fps"] not in (25, 50) or not episodes:
        raise ValueError("invalid recording clock or empty dataset")
    if [e["index"] for e in episodes] != list(range(len(episodes))):
        raise ValueError("episode indices must be consecutive")
    for key, success in (("successful_episodes", True), ("failure_episodes", False)):
        if selection.get(key) != [e["index"] for e in episodes if e["success"] is success]:
            raise ValueError("selection disagrees with episode outcomes")
    if any(
        type(e["success"]) is not bool or type(e["frames"]) is not int or e["frames"] <= 0
        for e in episodes
    ):
        raise ValueError("invalid episode outcomes or lengths")
    if set(split) != set(GROUPS) or any(type(v) is not list for v in split.values()):
        raise ValueError("train, validation and test lists required")
    ids = [i for group in GROUPS for i in split[group]]
    if any(type(i) is not int for i in ids) or sorted(ids) != list(range(len(episodes))):
        raise ValueError("split must cover every episode exactly once")
    if not split["train"] or not split["validation"]:
        raise ValueError("non-empty training and validation groups required")
    if not set(split["test"]) <= set(selection.get("independent_test_episodes", [])):
        raise ValueError("previously inspected episodes cannot become independent test data")
    scenes = {}
    for group in GROUPS:
        for index in split[group]:
            # Conservative grouping: even different disturbances at the same initial
            # position stay together. Names and outcome labels never define a scene.
            x = float(episodes[index]["scenario"]["x"])
            if not np.isfinite(x):
                raise ValueError("invalid scenario position")
            key = round(x, 9)
            if key in scenes and scenes[key] != group:
                raise ValueError("same initial position appears across splits")
            scenes[key] = group
    return manifest


def prepare(root, output, split, history=4, horizon=8):
    """Write small frame references, never duplicate or normalize held-out images."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite index: {output}")
    if any(type(v) is not int or v < 1 for v in (history, horizon)):
        raise ValueError("history and horizon must be positive integers")
    manifest = check_source(root, split)
    before = source_digest(root)
    dataset = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav")
    offsets = np.cumsum([0] + [e["frames"] for e in manifest["episodes"]])
    if len(dataset) != offsets[-1]:
        raise ValueError("dataset frame count mismatch")
    windows = {g: {"imitation": [], "outcome": []} for g in GROUPS}
    for group in GROUPS:
        # Test frames are not accessed during preparation. Evaluation gets a
        # separate, explicit data access step after the model is frozen.
        if group == "test":
            continue
        for ep in split[group]:
            episode = manifest["episodes"][ep]
            count = episode["frames"]
            phases = []
            for frame in range(count):
                row = dataset[int(offsets[ep]) + frame]
                if (
                    row["episode_index"].item() != ep
                    or row["frame_index"].item() != frame
                    or abs(row["timestamp"].item() - frame / manifest["fps"]) > 1e-4
                ):
                    raise ValueError("frame boundary or clock mismatch")
                phases.append(int(row["phase"].item()))
            active = {
                manifest["phase_names"].index(p) for p in ("running", "checking_grasp", "returning")
            }
            for end in range(history - 1, count):
                ref = [ep, end]
                windows[group]["outcome"].append(ref)
                # Exclude failures, terminal settling and chunks crossing completion.
                if (
                    episode["success"]
                    and end + horizon <= count
                    and all(p in active for p in phases[end : end + horizon])
                ):
                    windows[group]["imitation"].append(ref)
    if any(not windows[g]["imitation"] for g in ("train", "validation")):
        raise ValueError("training and validation need successful active action windows")
    if source_digest(root) != before:
        raise ValueError("source changed while preparing windows")
    report = {
        "schema_version": 1,
        "dataset_root": str(root),
        "source_sha256": before,
        "history": history,
        "horizon": horizon,
        "fps": manifest["fps"],
        "input_keys": list(INPUT_KEYS),
        "split": split,
        "windows": windows,
        "counts": {g: {k: len(v) for k, v in windows[g].items()} for g in GROUPS},
        "test_frames_loaded": False,
        "independent_evaluation_available": bool(split["test"]),
        "purpose": "training data preparation only; no model trained or evaluated",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


class TemporalDataset:
    """Lazy training/validation view; labels are separated from causal inputs.

    Outcome targets describe recorded controller outcomes, including external user
    stops. They are not a ground-truth autonomous stop decision.
    """

    def __init__(self, index_path, group="train", objective="imitation"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if group not in ("train", "validation") or objective not in ("imitation", "outcome"):
            raise ValueError("choose train/validation and imitation/outcome")
        self.index = read_json(index_path)
        root = Path(self.index["dataset_root"])
        self.manifest = check_source(root, self.index["split"])
        if source_digest(root) != self.index["source_sha256"]:
            raise ValueError("source changed; replay validation and preparation required")
        self.dataset = LeRobotDataset(self.manifest["repo_id"], root=root, video_backend="pyav")
        self.offsets = np.cumsum([0] + [e["frames"] for e in self.manifest["episodes"]])
        self.refs = self.index["windows"][group][objective]
        self.group, self.objective = group, objective

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, item):
        ep, end = self.refs[item]
        history, horizon = self.index["history"], self.index["horizon"]
        episode = self.manifest["episodes"][ep]
        if (
            ep not in self.index["split"][self.group]
            or end < history - 1
            or end >= episode["frames"]
        ):
            raise ValueError("invalid window reference")
        offset = int(self.offsets[ep])
        rows = [self.dataset[offset + i] for i in range(end - history + 1, end + 1)]
        inputs = {key: np.stack([r[key].numpy() for r in rows]) for key in INPUT_KEYS}
        # LeRobot squeezes singleton bool features to scalars on readback.
        if inputs["observation.holding"].shape == (history,):
            inputs["observation.holding"] = inputs["observation.holding"][:, None]
        expected = {
            IMAGE_KEY: (history, 3, 240, 320),
            "observation.state": (history, 5),
            "observation.holding": (history, 1),
        }
        if any(v.shape != expected[k] or not np.isfinite(v).all() for k, v in inputs.items()):
            raise ValueError("invalid temporal observation")
        if self.objective == "imitation":
            if not episode["success"] or end + horizon > episode["frames"]:
                raise ValueError("invalid imitation target")
            targets = [self.dataset[offset + i] for i in range(end, end + horizon)]
            active = {
                self.manifest["phase_names"].index(p)
                for p in ("running", "checking_grasp", "returning")
            }
            if any(int(r["phase"].item()) not in active for r in targets):
                raise ValueError("imitation target crosses terminal state")
            target = {"action": np.stack([r["action"].numpy() for r in targets])}
            if target["action"].shape != (horizon, 5) or not np.isfinite(target["action"]).all():
                raise ValueError("invalid action target")
        else:
            target = {"episode_success": np.array([episode["success"]], dtype=bool)}
        return {"inputs": inputs, "targets": target, "episode": ep, "frame": end}
