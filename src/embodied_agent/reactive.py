"""Small causal feedback BC baseline, with hand-designed RGB features.

No object pose, clock, phase label, recovery counter or IK is a policy input.
This is not ACT or an end-to-end vision network.
"""

import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from .temporal_data import INPUT_KEYS, TemporalDataset, read_json, source_digest

FEATURE_VERSION = "red-moments-joints-contact-v1"
FEATURE_COUNT = 12


def observation_features(rgb, joints, holding):
    rgb, joints = np.asarray(rgb), np.asarray(joints, dtype=np.float32)
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError("expected uint8 RGB image at 240x320")
    if joints.shape != (5,) or not np.isfinite(joints).all():
        raise ValueError("invalid joint observation")
    contact = np.asarray(holding)
    if contact.size != 1 or contact.item() not in (False, True):
        raise ValueError("invalid contact observation")
    red, green, blue = rgb.astype(np.float32).transpose(2, 0, 1)
    y, x = np.nonzero((red > 70) & (red > 1.6 * green) & (red > 1.4 * blue))
    # Unlike the first-frame locator, a reactive observation must represent
    # occlusion explicitly rather than invent a pose or reject every covered cube.
    visual = np.zeros(6, dtype=np.float32)
    if len(x):
        visual[:] = [
            (x.mean() + 0.5) / 320,
            (y.mean() + 0.5) / 240,
            len(x) / 76800,
            np.std(x) / 320,
            np.std(y) / 240,
            1,
        ]
    return np.concatenate((visual, joints, [float(contact.item())])).astype(np.float32)


def network(history, horizon, hidden):
    from torch import nn

    return nn.Sequential(
        nn.Linear(history * FEATURE_COUNT, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, horizon * 5),
    )


def training_arrays(index_path, group):
    """Load only referenced successful train/validation frames, once per frame."""
    view = TemporalDataset(index_path, group)
    history, horizon = view.index["history"], view.index["horizon"]
    features, actions, phases = {}, {}, {}
    active = {
        view.manifest["phase_names"].index(p) for p in ("running", "checking_grasp", "returning")
    }
    needed = set()
    for ep, end in view.refs:
        if (
            ep not in view.index["split"][group]
            or not view.manifest["episodes"][ep]["success"]
            or end < history - 1
            or end + horizon > view.manifest["episodes"][ep]["frames"]
        ):
            raise ValueError("invalid successful training reference")
        needed.update((ep, t) for t in range(end - history + 1, end + horizon))
    for ep, t in sorted(needed):
        row = view.dataset[int(view.offsets[ep]) + t]
        if (
            row["episode_index"].item() != ep
            or row["frame_index"].item() != t
            or abs(row["timestamp"].item() - t / view.manifest["fps"]) > 1e-4
        ):
            raise ValueError("training frame boundary or timestamp mismatch")
        rgb = row[INPUT_KEYS[0]].numpy()
        if (
            rgb.shape != (3, 240, 320)
            or not np.isfinite(rgb).all()
            or rgb.min() < 0
            or rgb.max() > 1
        ):
            raise ValueError("invalid training image")
        rgb = np.rint(rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
        features[ep, t] = observation_features(
            rgb, row[INPUT_KEYS[1]].numpy(), row[INPUT_KEYS[2]].numpy()
        )
        action = row["action"].numpy()
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("invalid training action")
        actions[ep, t] = action
        phases[ep, t] = row["phase"].item()
    x, y, joints = [], [], []
    for ep, end in view.refs:
        if any(phases[ep, t] not in active for t in range(end, end + horizon)):
            raise ValueError("action target crosses terminal phase")
        x.append(np.concatenate([features[ep, t] for t in range(end - history + 1, end + 1)]))
        y.append(np.stack([actions[ep, t] for t in range(end, end + horizon)]))
        joints.append(features[ep, end][6:11])
    if not x:
        raise ValueError("no successful action windows")
    return np.stack(x), np.stack(y), np.stack(joints), view


def check_experiment(config, manifest):
    if config["development"] != [e["scenario"] for e in manifest["episodes"]]:
        raise ValueError("development recordings differ from frozen experiment")
    seen = {round(float(e["x"]), 9) for e in config["development"]}
    cases = config["test"]
    if not cases or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("unique reserved test scenarios required")
    if any(round(float(c["x"]), 9) in seen for c in cases):
        raise ValueError("test position overlaps development")
    if not 1 <= config["epochs"] <= 5000 or not 16 <= config["hidden"] <= 512:
        raise ValueError("invalid training budget")


def train(index_path, experiment_path, output):
    import torch

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite model: {output}")
    config = read_json(experiment_path)
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    train_x, train_y, train_joints, view = training_arrays(index_path, "train")
    check_experiment(config, view.manifest)
    if config["split"] != view.index["split"]:
        raise ValueError("experiment and index split disagree")
    val_x, val_y, val_joints, _ = training_arrays(index_path, "validation")
    # Train-only normalization. Test frames and outcomes are never read.
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-3)
    from .physics import PhysicsWorld

    world = PhysicsWorld()
    try:
        bounds = world.model.actuator_ctrlrange.copy()
    finally:
        world.close()
    ranges = (bounds[:, 1] - bounds[:, 0]).astype(np.float32)

    def tensors(x, y, joints):
        return (
            torch.from_numpy(((x - mean) / scale).astype(np.float32)),
            torch.from_numpy(((y - joints[:, None]) / ranges).astype(np.float32)).flatten(1),
        )

    x, y = tensors(train_x, train_y, train_joints)
    vx, vy = tensors(val_x, val_y, val_joints)
    history, horizon = view.index["history"], view.index["horizon"]
    model = network(history, horizon, config["hidden"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    best, best_epoch, weights, curve = float("inf"), None, None, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for ids in torch.randperm(len(x)).split(128):
            optimizer.zero_grad()
            loss = (model(x[ids]) - y[ids]).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float((model(vx) - vy).square().mean())
            train_loss = float((model(x) - y).square().mean())
        if not np.isfinite([score, train_loss]).all():
            raise ValueError("non-finite training loss")
        if score < best:
            best, best_epoch = score, epoch
            weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0:
            row = {"epoch": epoch, "train_mse": train_loss, "validation_mse": score}
            curve.append(row)
            print(json.dumps(row), flush=True)
    if source_digest(Path(view.index["dataset_root"])) != view.index["source_sha256"]:
        raise ValueError("dataset changed during training")
    output.mkdir(parents=True)
    torch.save(weights, output / "weights.pt")
    np.savez_compressed(output / "normalization.npz", mean=mean, scale=scale, bounds=bounds)
    experiment_bytes = Path(experiment_path).read_bytes()
    (output / "experiment.json").write_bytes(experiment_bytes)
    metadata = {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "algorithm": "causal RGB-moment and proprioception MLP action-chunk BC",
        "control": "recompute every observation; execute first action of predicted chunk",
        "history": history,
        "horizon": horizon,
        "hidden": config["hidden"],
        "fps": view.index["fps"],
        "selected_epoch": best_epoch,
        "validation_normalized_mse": best,
        "hold_position_validation_normalized_mse": float(vy.square().mean()),
        "train_windows": len(x),
        "validation_windows": len(vx),
        "curve": curve,
        "source_sha256": view.index["source_sha256"],
        "split": view.index["split"],
        "model_sha256": view.manifest["model_sha256"],
        "experiment_sha256": hashlib.sha256(experiment_bytes).hexdigest(),
        "weights_sha256": hashlib.sha256((output / "weights.pt").read_bytes()).hexdigest(),
        "normalization_sha256": hashlib.sha256(
            (output / "normalization.npz").read_bytes()
        ).hexdigest(),
        "input_keys": list(INPUT_KEYS),
        "test_used_for_training": False,
        "limitations": "fixed red cube/camera; hand-designed vision; no learned completion or stop classifier",
    }
    (output / "training.json").write_text(json.dumps(metadata, indent=2))
    return metadata


class ReactivePolicy:
    def __init__(self, root):
        import torch

        root = Path(root)
        self.metadata = read_json(root / "training.json")
        m = self.metadata
        if m["feature_version"] != FEATURE_VERSION or m["fps"] not in (25, 50):
            raise ValueError("unsupported reactive model")
        if (
            type(m["history"]) is not int
            or not 1 <= m["history"] <= 128
            or type(m["horizon"]) is not int
            or not 1 <= m["horizon"] <= 128
            or type(m["hidden"]) is not int
            or not 16 <= m["hidden"] <= 512
        ):
            raise ValueError("invalid model dimensions")
        for file, key in (
            ("weights.pt", "weights_sha256"),
            ("normalization.npz", "normalization_sha256"),
            ("experiment.json", "experiment_sha256"),
        ):
            if hashlib.sha256((root / file).read_bytes()).hexdigest() != m[key]:
                raise ValueError("checkpoint integrity mismatch")
        with np.load(root / "normalization.npz", allow_pickle=False) as data:
            self.mean, self.scale, self.bounds = (
                data[k].copy() for k in ("mean", "scale", "bounds")
            )
        if (
            self.mean.shape != (m["history"] * FEATURE_COUNT,)
            or self.scale.shape != self.mean.shape
            or self.bounds.shape != (5, 2)
            or not all(np.isfinite(a).all() for a in (self.mean, self.scale, self.bounds))
            or np.any(self.scale <= 0)
            or np.any(self.bounds[:, 0] >= self.bounds[:, 1])
        ):
            raise ValueError("invalid normalization or bounds")
        self.model = network(m["history"], m["horizon"], m["hidden"])
        self.model.load_state_dict(
            torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
        )
        self.model.eval()
        if any(not torch.isfinite(v).all() for v in self.model.parameters()):
            raise ValueError("non-finite model weights")

    def predict(self, history):
        import torch

        history = np.asarray(history, dtype=np.float32)
        if (
            history.shape != (self.metadata["history"], FEATURE_COUNT)
            or not np.isfinite(history).all()
        ):
            raise ValueError("invalid feature history")
        normalized = ((history.flatten() - self.mean) / self.scale).astype(np.float32)
        with torch.no_grad():
            delta = self.model(torch.from_numpy(normalized)[None]).numpy().reshape(-1, 5)
        action = history[-1, 6:11] + delta * (self.bounds[:, 1] - self.bounds[:, 0])
        if not np.isfinite(action).all():
            raise ValueError("non-finite policy action")
        return np.clip(action, self.bounds[:, 0], self.bounds[:, 1])


class ReactiveExecutor:
    def __init__(self, policy, initial_joints, max_seconds=24):
        self.policy = policy
        self.fps = policy.metadata["fps"]
        self.history = deque(maxlen=policy.metadata["history"])
        self.command = np.asarray(initial_joints, dtype=float).copy()
        if self.command.shape != (5,) or not np.isfinite(self.command).all():
            raise ValueError("invalid initial joints")
        self.command = np.clip(self.command, policy.bounds[:, 0], policy.bounds[:, 1])
        self.limit = np.array([2.0, 2.0, 2.0, 1.0, 1.0]) / self.fps
        self.max_ticks = round(max_seconds * self.fps)
        if self.max_ticks < 1 or not np.isfinite(max_seconds):
            raise ValueError("invalid execution budget")
        self.ticks, self.state, self.reason = 0, "running", ""

    def stop(self, joints, reason="user stop"):
        if self.state != "stopped":
            joints = np.asarray(joints)
            if joints.shape == (5,) and np.isfinite(joints).all():
                self.command[:3] = np.clip(
                    joints[:3], self.policy.bounds[:3, 0], self.policy.bounds[:3, 1]
                )
            self.state, self.reason = "stopped", reason
        return self.command.copy()

    def step(self, rgb, joints, holding, stop=False):
        if self.state == "stopped":
            return self.command.copy()
        if stop:
            return self.stop(joints)
        if self.ticks >= self.max_ticks:
            return self.stop(joints, "execution timeout")
        self.ticks += 1
        try:
            self.history.append(observation_features(rgb, joints, holding))
            # Hold still to acquire real history; do not inject future/padded frames.
            if len(self.history) < self.history.maxlen:
                return self.command.copy()
            action = self.policy.predict(np.stack(self.history))[0]
            action = np.asarray(action)
            if action.shape != (5,) or not np.isfinite(action).all():
                raise ValueError("invalid predicted action")
            action = np.clip(action, self.policy.bounds[:, 0], self.policy.bounds[:, 1])
            self.command += np.clip(action - self.command, -self.limit, self.limit)
        except (ValueError, RuntimeError) as exc:
            return self.stop(joints, f"invalid policy observation or output: {exc}")
        return self.command.copy()
