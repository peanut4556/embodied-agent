"""Bounded sensitivity checks for candidate normals; not an angle certificate."""

import numpy as np


def normal_stability(points, indices):
    points = np.asarray(points, dtype=float)
    idx = np.asarray(indices)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or idx.ndim != 1
        or not np.issubdtype(idx.dtype, np.integer)
        or len(idx) < 3
        or len(np.unique(idx)) != len(idx)
        or np.any(idx < 0)
        or np.any(idx >= len(points))
    ):
        raise ValueError("finite points and unique valid group indices required")
    selected = points[idx]
    center = selected.mean(0)
    _, _, axes = np.linalg.svd(selected - center, full_matrices=False)
    reference = axes[-1]
    views = []

    def score(name, ids):
        cloud = points[ids]
        row = {"name": name, "points": len(ids)}
        if len(ids) < 20:
            return row | {"status": "unknown", "reason": "insufficient_points"}
        _, _, basis = np.linalg.svd(cloud - cloud.mean(0), full_matrices=False)
        spans = np.ptp((cloud - cloud.mean(0)) @ basis[:2].T, axis=0)
        if min(spans) < 0.005:
            return row | {"status": "unknown", "reason": "insufficient_span"}
        angle = float(np.rad2deg(np.arccos(np.clip(abs(basis[-1] @ reference), 0, 1))))
        return row | {
            "status": "measured",
            "normal": basis[-1].tolist(),
            "angle_degrees": angle,
            "minimum_span_m": float(min(spans)),
        }

    for axis in range(2):
        coordinates = (selected - center) @ axes[axis]
        split = np.median(coordinates)
        for side, keep in (("low", coordinates <= split), ("high", coordinates > split)):
            views.append(score(f"spatial_axis_{axis}_{side}", idx[keep]))
    # Re-select from ALL reliable points, not just the already selected group;
    # this explicitly probes sequential RANSAC's inlier-selection bias.
    residual = np.abs((points - center) @ reference)
    for threshold in (0.001, 0.002):
        ids = np.flatnonzero(residual <= threshold)
        row = score(f"all_points_residual_{threshold:.3f}", ids)
        row["jaccard_with_original"] = float(
            len(np.intersect1d(idx, ids)) / len(np.union1d(idx, ids))
        )
        views.append(row)
    angles = [v["angle_degrees"] for v in views if v["status"] == "measured"]
    unstable = any(a > 5 for a in angles)
    return {
        "status": "unstable" if unstable else "unknown" if len(angles) != 6 else "stable_candidate",
        "max_change_degrees": max(angles, default=None),
        "views": views,
        "angle_bound_certified": False,
        "clearance_certified": False,
    }
