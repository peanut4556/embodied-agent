"""Coordinate-only split before plane fitting; independent proposals on each side."""

import numpy as np

from .plane_groups import plane_groups
from .rgbd_filter import erode_mask


def pixel_partitions(shape):
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("positive image dimensions required")
    y, x = np.indices(shape)
    a = ((y // 8 + x // 8) % 2) == 0
    # A one-pixel guard also separates the 4-neighbor filter dependencies.
    return erode_mask(a), erode_mask(~a)


def heldout_groups(points, reliable_mask):
    points, mask = np.asarray(points), np.asarray(reliable_mask, dtype=bool)
    if points.shape != (*mask.shape, 3) or mask.ndim != 2 or not np.isfinite(points[mask]).all():
        raise ValueError("finite masked image points required")
    a, b = pixel_partitions(mask.shape)
    flat_indices = np.full(mask.shape, -1, dtype=int)
    flat_indices[mask] = np.arange(mask.sum())
    train_ids, validation_ids = flat_indices[mask & a], flat_indices[mask & b]
    reliable = points[mask]
    train = plane_groups(reliable[train_ids])
    validation = plane_groups(reliable[validation_ids])
    candidates = [g for g in train["groups"] if g["proposal_accepted"]]
    validators = [g for g in validation["groups"] if g["proposal_accepted"]]
    distances = np.array(
        [
            [np.linalg.norm(np.array(g["center"]) - h["center"]) for h in validators]
            for g in candidates
        ]
    )
    rows = []
    for i, g in enumerate(candidates):
        verdict = {"status": "unknown", "reason": "no_unique_spatial_match"}
        if validators:
            j = int(np.argmin(distances[i]))
            reciprocal = int(np.argmin(distances[:, j])) == i
            # Competing similarly close groups remain ambiguous, rather than
            # choosing the one whose direction happens to agree.
            row_unique = sum(distances[i] <= distances[i, j] + 0.002) == 1
            col_unique = sum(distances[:, j] <= distances[i, j] + 0.002) == 1
            if reciprocal and row_unique and col_unique and distances[i, j] <= 0.02:
                h = validators[j]
                n, m = np.array(g["normal"]), np.array(h["normal"])
                delta = np.array(g["center"]) - h["center"]
                angle = float(np.rad2deg(np.arccos(np.clip(abs(n @ m), 0, 1))))
                offset = float(max(abs(delta @ n), abs(delta @ m)))
                passed = angle <= 5 and offset <= 0.003
                verdict = {
                    "status": "consistent_candidate" if passed else "inconsistent",
                    "validation_group": j,
                    "center_distance_m": float(distances[i, j]),
                    "angle_degrees": angle,
                    "plane_offset_m": offset,
                    "validation_points": len(h["indices"]),
                }
        rows.append({**g, "indices": train_ids[g["indices"]].tolist(), "holdout": verdict})
    return {
        "groups": rows,
        "candidate_pixels": len(train_ids),
        "validation_pixels": len(validation_ids),
        "guard_or_unused_pixels": int(mask.sum()) - len(train_ids) - len(validation_ids),
        "validation_groups": len(validators),
        "status": "candidate_groups" if rows else "unknown",
        "angle_bound_certified": False,
    }
