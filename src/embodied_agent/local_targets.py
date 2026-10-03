"""Strict training-only local-action supervision alongside unchanged expert BC."""

from pathlib import Path

import numpy as np
import torch

from .correction_data import payload_digest
from .memory_policy import digest
from .temporal_data import read_json


def auxiliary_batch(episodes, y, policy, config, correction_roots):
    if not config:
        return None
    if set(config) != {"path", "sha256", "weight"} or not 0 <= config["weight"] <= 1:
        raise ValueError("invalid local supervision config")
    path = Path(config["path"])
    if digest(path) != config["sha256"]:
        raise ValueError("local target integrity mismatch")
    report = read_json(path)
    if (
        report["status"] != "complete"
        or report["weights_sha256"] != policy.metadata["weights_sha256"]
        or len(correction_roots) != 1
        or report["source_sha256"] != payload_digest(correction_roots[0])
    ):
        raise ValueError("local target provenance mismatch")
    mapping = {e["episode"]: i for i, e in enumerate(episodes) if e.get("is_correction")}
    targets = y.clone()
    mask = torch.zeros((*y.shape[:2], 1), dtype=y.dtype)
    seen = set()
    rejected = 0
    for row in report["rows"]:
        ep, t = row["episode"], row["tick"]
        if row["group"] != "train" or ep not in mapping or (ep, t) in seen:
            raise ValueError("local targets must be unique training-only correction frames")
        seen.add((ep, t))
        i = mapping[ep]
        episode = episodes[i]
        if type(t) is not int or not 0 <= t < len(episode["x"]) or not episode["loss_mask"][t]:
            raise ValueError("local target points to unsupervised frame")
        a = np.asarray(row["projected_command"])
        if (
            a.shape != (5,)
            or not np.isfinite(a).all()
            or np.any(a < policy.bounds[:, 0])
            or np.any(a > policy.bounds[:, 1])
            or not np.isfinite([row["loss_before"], row["loss_after"]]).all()
            or min(row["loss_before"], row["loss_after"]) < 0
            or not np.isfinite(row["target_replay_max_error"])
            or not 0 <= row["target_replay_max_error"] <= 1e-10
        ):
            raise ValueError("invalid local target or state replay")
        if row["loss_after"] >= row["loss_before"] - 1e-12:
            rejected += 1
            continue
        targets[i, t] = torch.as_tensor(
            (a - policy.bounds[:, 0]) / (policy.bounds[:, 1] - policy.bounds[:, 0]), dtype=y.dtype
        )
        mask[i, t] = 1
    if not mask.any():
        raise ValueError("no accepted local targets")
    return (
        targets,
        mask,
        {
            "accepted": int(mask.sum()),
            "rejected": rejected,
            "sha256": config["sha256"],
            "weight": config["weight"],
        },
    )
