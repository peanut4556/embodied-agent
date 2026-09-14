"""Causal GRU behavior cloning with independent per-execution recurrent state."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .reactive import FEATURE_COUNT, FEATURE_VERSION, observation_features
from .temporal_data import INPUT_KEYS, check_source, read_json, source_digest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_network(hidden):
    import torch

    class SequenceNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(FEATURE_COUNT, hidden, batch_first=True)
            self.head = torch.nn.Linear(hidden, 5)

        def forward(self, observations, state=None):
            encoded, state = self.gru(observations, state)
            return self.head(encoded), state

    return SequenceNetwork()


def check_config(config, manifest):
    epochs = config["candidate_epochs"]
    if (
        not epochs
        or any(type(e) is not int or not 1 <= e <= 3000 for e in epochs)
        or epochs != sorted(set(epochs))
        or type(config["hidden"]) is not int
        or not 16 <= config["hidden"] <= 256
        or not 0 < config["learning_rate"] < 1
        or not 1 <= config["max_seconds"] <= 60
    ):
        raise ValueError("invalid memory experiment budget")
    if config["split"]["test"]:
        raise ValueError("memory training takes development data only")
    seen = {round(e["scenario"]["x"], 9) for e in manifest["episodes"]}
    seen.update(round(x, 9) for x in config["previous_test_positions"])
    names = [c["name"] for c in config["test"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("unique test case names required")
    if any(round(c["x"], 9) in seen for c in config["test"]):
        raise ValueError("reserved test scene overlaps inspected positions")
    # Physical selection must use the existing validation scenarios, never training
    # scenes or reserved test scenes. Names may differ; fixture parameters may not.
    clean = lambda case: {k: v for k, v in case.items() if k != "name"}
    expected = [clean(manifest["episodes"][i]["scenario"]) for i in config["split"]["validation"]]
    if [clean(c) for c in config["validation_cases"]] != expected:
        raise ValueError("physical selection cases must match development validation")


def sequences(root, split, group):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if group not in ("train", "validation"):
        raise ValueError("only development groups may be loaded for training")
    root = Path(root)
    manifest = check_source(root, split)
    dataset = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav")
    offsets = np.cumsum([0] + [e["frames"] for e in manifest["episodes"]])
    if len(dataset) != offsets[-1]:
        raise ValueError("dataset length mismatch")
    active = {manifest["phase_names"].index(p) for p in ("running", "checking_grasp", "returning")}
    episodes = []
    for ep in split[group]:
        record = manifest["episodes"][ep]
        if not record["success"]:
            continue
        observations, actions = [], []
        terminal_seen = False
        for frame in range(record["frames"]):
            row = dataset[int(offsets[ep]) + frame]
            if (
                row["episode_index"].item() != ep
                or row["frame_index"].item() != frame
                or abs(row["timestamp"].item() - frame / manifest["fps"]) > 1e-4
            ):
                raise ValueError("sequence boundary or timestamp mismatch")
            is_active = row["phase"].item() in active
            if not is_active:
                terminal_seen = True
                continue
            if terminal_seen:
                raise ValueError("active frames after terminal state")
            image = row[INPUT_KEYS[0]].numpy()
            if (
                image.shape != (3, 240, 320)
                or not np.isfinite(image).all()
                or image.min() < 0
                or image.max() > 1
            ):
                raise ValueError("invalid sequence image")
            rgb = np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
            observations.append(
                observation_features(rgb, row[INPUT_KEYS[1]].numpy(), row[INPUT_KEYS[2]].numpy())
            )
            action = row["action"].numpy()
            if action.shape != (5,) or not np.isfinite(action).all():
                raise ValueError("invalid sequence target")
            actions.append(action)
        if not actions:
            raise ValueError("successful episode has no active frames")
        episodes.append({"episode": ep, "x": np.stack(observations), "y": np.stack(actions)})
    if not episodes:
        raise ValueError("no successful sequences")
    return episodes


def padded_batch(episodes, mean, scale, bounds):
    import torch

    length = max(len(e["x"]) for e in episodes)
    x = np.zeros((len(episodes), length, FEATURE_COUNT), dtype=np.float32)
    y = np.zeros((len(episodes), length, 5), dtype=np.float32)
    mask = np.zeros((len(episodes), length, 1), dtype=np.float32)
    for i, episode in enumerate(episodes):
        n = len(episode["x"])
        x[i, :n] = (episode["x"] - mean) / scale
        y[i, :n] = (episode["y"] - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        mask[i, :n] = 1
    return tuple(torch.from_numpy(v) for v in (x, y, mask))


def masked_loss(prediction, targets, mask):
    return ((prediction - targets).square() * mask).sum() / (mask.sum() * targets.shape[-1])


def train(root, experiment_path, output):
    import torch

    from .physics import PhysicsWorld

    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite run: {output}")
    experiment_bytes = Path(experiment_path).read_bytes()
    config = json.loads(experiment_bytes)
    manifest = check_source(root, config["split"])
    check_config(config, manifest)
    source_hash = source_digest(root)
    training = sequences(root, config["split"], "train")
    validation = sequences(root, config["split"], "validation")
    all_train = np.concatenate([e["x"] for e in training])
    mean, scale = all_train.mean(0), np.maximum(all_train.std(0), 1e-3)
    world = PhysicsWorld()
    try:
        bounds = world.model.actuator_ctrlrange.copy()
    finally:
        world.close()
    for episode in training + validation:
        if np.any(episode["y"] < bounds[:, 0] - 1e-6) or np.any(episode["y"] > bounds[:, 1] + 1e-6):
            raise ValueError("training targets exceed actuator bounds")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    model = make_network(config["hidden"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    x, y, mask = padded_batch(training, mean, scale, bounds)
    vx, vy, vm = padded_batch(validation, mean, scale, bounds)
    output.mkdir(parents=True)
    (output / "experiment.json").write_bytes(experiment_bytes)
    status_path = output / "run.json"
    report = {
        "status": "training",
        "source_sha256": source_hash,
        "experiment_sha256": hashlib.sha256(experiment_bytes).hexdigest(),
        "train_episodes": [e["episode"] for e in training],
        "validation_episodes": [e["episode"] for e in validation],
        "train_frames": sum(len(e["x"]) for e in training),
        "validation_frames": sum(len(e["x"]) for e in validation),
        "test_used_for_training": False,
        "candidates": [],
        "curve": [],
    }
    status_path.write_text(json.dumps(report, indent=2))
    try:
        for epoch in range(1, max(config["candidate_epochs"]) + 1):
            model.train()
            optimizer.zero_grad()
            prediction, _ = model(x)  # Zero hidden state for each separate episode.
            loss = masked_loss(prediction, y, mask)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if epoch == 1 or epoch % 50 == 0 or epoch in config["candidate_epochs"]:
                model.eval()
                with torch.no_grad():
                    val_loss = float(masked_loss(model(vx)[0], vy, vm))
                row = {
                    "epoch": epoch,
                    "train_mse_before_update": float(loss.detach()),
                    "validation_mse": val_loss,
                }
                report["curve"].append(row)
                print(json.dumps(row), flush=True)
            if epoch in config["candidate_epochs"]:
                target = output / f"epoch-{epoch}"
                target.mkdir()
                torch.save(model.state_dict(), target / "weights.pt")
                np.savez_compressed(
                    target / "normalization.npz", mean=mean, scale=scale, bounds=bounds
                )
                (target / "experiment.json").write_bytes(experiment_bytes)
                metadata = {
                    "schema_version": 1,
                    "algorithm": "causal GRU absolute-action BC",
                    "feature_version": FEATURE_VERSION,
                    "history": 1,
                    "horizon": 1,
                    "memory": "persistent hidden state across the episode; reset for every execution",
                    "hidden": config["hidden"],
                    "fps": manifest["fps"],
                    "epoch": epoch,
                    "validation_normalized_mse": val_loss,
                    "source_sha256": source_hash,
                    "model_sha256": manifest["model_sha256"],
                    "input_keys": list(INPUT_KEYS),
                    "feature_source_sha256": digest(Path(__file__).with_name("reactive.py")),
                    "policy_source_sha256": digest(__file__),
                    "experiment_sha256": report["experiment_sha256"],
                    "weights_sha256": digest(target / "weights.pt"),
                    "normalization_sha256": digest(target / "normalization.npz"),
                    "test_used_for_training": False,
                }
                (target / "training.json").write_text(json.dumps(metadata, indent=2))
                report["candidates"].append(
                    {
                        "directory": target.name,
                        "epoch": epoch,
                        "validation_mse": val_loss,
                        "weights_sha256": metadata["weights_sha256"],
                    }
                )
                status_path.write_text(json.dumps(report, indent=2))
        if source_digest(root) != source_hash:
            raise ValueError("source dataset changed during training")
        report["status"] = "trained"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        status_path.write_text(json.dumps(report, indent=2))
    return report


class MemoryPolicy:
    def __init__(self, root):
        import torch

        root = Path(root)
        self.metadata = read_json(root / "training.json")
        m = self.metadata
        if (
            m["algorithm"] != "causal GRU absolute-action BC"
            or m["feature_version"] != FEATURE_VERSION
            or m["fps"] not in (25, 50)
            or type(m["hidden"]) is not int
            or not 16 <= m["hidden"] <= 256
            or m["history"] != 1
            or m["horizon"] != 1
        ):
            raise ValueError("unsupported memory checkpoint")
        for file, key in (
            ("weights.pt", "weights_sha256"),
            ("normalization.npz", "normalization_sha256"),
            ("experiment.json", "experiment_sha256"),
        ):
            if digest(root / file) != m[key]:
                raise ValueError("memory checkpoint integrity mismatch")
        with np.load(root / "normalization.npz", allow_pickle=False) as data:
            self.mean, self.scale, self.bounds = (
                data[k].copy() for k in ("mean", "scale", "bounds")
            )
        if (
            self.mean.shape != (FEATURE_COUNT,)
            or self.scale.shape != self.mean.shape
            or self.bounds.shape != (5, 2)
            or np.any(self.scale <= 0)
            or np.any(self.bounds[:, 0] >= self.bounds[:, 1])
            or not all(np.isfinite(v).all() for v in (self.mean, self.scale, self.bounds))
        ):
            raise ValueError("invalid memory normalization")
        self.model = make_network(m["hidden"])
        self.model.load_state_dict(
            torch.load(root / "weights.pt", weights_only=True, map_location="cpu")
        )
        self.model.eval()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("non-finite memory weights")

    def session(self):
        return MemorySession(self)


class MemorySession:
    def __init__(self, policy):
        self.policy = policy
        self.metadata, self.bounds = policy.metadata, policy.bounds
        self.hidden = None

    def predict(self, observations):
        import torch

        observations = np.asarray(observations, dtype=np.float32)
        if observations.shape != (1, FEATURE_COUNT) or not np.isfinite(observations).all():
            raise ValueError("invalid memory observation")
        normalized = ((observations - self.policy.mean) / self.policy.scale).astype(np.float32)
        with torch.no_grad():
            prediction, hidden = self.policy.model(torch.from_numpy(normalized)[None], self.hidden)
        action = (
            prediction.numpy().reshape(1, 5) * (self.bounds[:, 1] - self.bounds[:, 0])
            + self.bounds[:, 0]
        )
        if not np.isfinite(action).all() or not torch.isfinite(hidden).all():
            raise ValueError("non-finite memory prediction")
        self.hidden = hidden
        return np.clip(action, self.bounds[:, 0], self.bounds[:, 1])
