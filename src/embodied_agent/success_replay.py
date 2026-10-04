"""Training-only distillation on verified successful parent rollout observations."""

import numpy as np

from .memory_policy import digest
from .temporal_data import read_json


def replay_batch(config, policy, train_positions, validation_positions):
    import torch

    if config is None:
        return None
    if set(config) != {"path", "sha256", "weight"} or not 0 <= config["weight"] <= 10:
        raise ValueError("invalid success replay configuration")
    if digest(config["path"]) != config["sha256"]:
        raise ValueError("success replay changed")
    pack = read_json(config["path"])
    if (
        pack["group"] != "train"
        or pack["weights_sha256"] != policy.metadata["weights_sha256"]
        or pack["normalization_sha256"] != policy.metadata["normalization_sha256"]
        or pack["model_sha256"] != policy.metadata["model_sha256"]
        or pack["fps"] != policy.metadata["fps"]
        or not pack["result"]["quality"]["verified_pick_place"]
        or pack["position"] not in train_positions
        or pack["position"] in validation_positions
    ):
        raise ValueError("replay must be a verified train-only rollout of the frozen parent")
    observations = np.asarray(pack["observations"], dtype=np.float32)
    targets = np.asarray(pack["normalized_targets"], dtype=np.float32)
    if (
        observations.ndim != 2
        or observations.shape[1] != 12
        or len(observations) == 0
        or targets.shape != (len(observations), 5)
        or not np.isfinite(observations).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("invalid success replay arrays")
    x = torch.from_numpy(((observations - policy.mean) / policy.scale).astype(np.float32))[None]
    y = torch.from_numpy(targets)[None]
    with torch.no_grad():
        if not torch.allclose(policy.model(x)[0], y, atol=1e-6, rtol=1e-6):
            raise ValueError("replay targets differ from the frozen parent")
    return x, y, {**config, "frames": len(observations), "position": pack["position"]}
