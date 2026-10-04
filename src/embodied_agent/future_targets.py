"""Training-only future outcome labels with verified-completion masking."""

from pathlib import Path

import numpy as np
import torch

from .correction_data import payload_digest
from .grasp_quality import GraspQuality
from .memory_policy import digest
from .temporal_data import read_json


def completion_flags(trace, fps=25):
    quality = GraspQuality(fps)
    flags = []
    for row in trace:
        quality.observe(row["holding"], row["block_xyz"], row["command"], row["inside_box"])
        flags.append(quality.result()["verified_pick_place"])
    return flags


def future_batch(episodes, x, policy, config, roots):
    if not config:
        return None
    if (
        set(config) != {"path", "sha256", "weight", "completion_mask"}
        or type(config["completion_mask"]) is not bool
        or not 0 <= config["weight"] <= 1
    ):
        raise ValueError("invalid future supervision config")
    if digest(config["path"]) != config["sha256"]:
        raise ValueError("future target hash mismatch")
    pack = read_json(Path(config["path"]))
    if (
        pack["group"] != "train"
        or len(roots) != 1
        or pack["source_sha256"] != payload_digest(roots[0])
        or pack["weights_sha256"] != policy.metadata["weights_sha256"]
    ):
        raise ValueError("future target provenance mismatch")
    horizons = pack["horizons"]
    if horizons != [1, 25, 100, 200, 300]:
        raise ValueError("frozen future horizons differ")
    labels = torch.zeros((*x.shape[:2], 25), dtype=x.dtype)
    mask = torch.zeros_like(labels)
    mapping = {e["episode"]: i for i, e in enumerate(episodes) if e.get("is_correction")}
    seen = set()
    protected = 0
    for row in pack["targets"]:
        ep, t, h = row["episode"], row["takeover_tick"], row["horizon_ticks"]
        if row["group"] != "train" or ep not in mapping or h not in horizons or (ep, h) in seen:
            raise ValueError("unique training future targets required")
        seen.add((ep, h))
        i = mapping[ep]
        if (
            type(t) is not int
            or not 0 <= t < len(episodes[i]["x"])
            or not episodes[i]["loss_mask"][t]
        ):
            raise ValueError("future anchor is not a supervised correction frame")
        values = np.array(
            row["expert_object_target_m"]
            + [float(row["expert_holding_target"]), float(row["expert_inside_box"])]
        )
        if (
            values.shape != (5,)
            or not np.isfinite(values).all()
            or type(row["autonomous_verified_complete"]) is not bool
        ):
            raise ValueError("invalid future target")
        start = 5 * horizons.index(h)
        labels[i, t, start : start + 5] = torch.as_tensor(values, dtype=x.dtype)
        skip = config["completion_mask"] and row["autonomous_verified_complete"]
        mask[i, t, start : start + 5] = not skip
        protected += bool(skip)
    if not mask.any():
        raise ValueError("no active future targets")
    return (
        labels,
        mask,
        {"targets": len(seen), "protected": protected, "active": int(mask.sum()) // 5, **config},
    )
