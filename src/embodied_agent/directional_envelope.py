"""Observation-aligned diameter slabs; no assumption that PCA recovers cube axes."""

import numpy as np

from .conservative_geometry import envelope


def directional_envelope(points, side_upper_m=0.05, point_error_m=0.003):
    points = np.asarray(points, dtype=float)
    baseline = envelope(points, side_upper_m, point_error_m)
    if len(points) < 30:
        raise ValueError("at least 30 reliable points required")
    _, _, frame = np.linalg.svd(points - points.mean(0), full_matrices=False)
    # These are observation directions, not a fitted object orientation. Even a
    # wrong direction remains valid: cube diameter bounds projection on ANY axis.
    projected = points @ frame.T
    radius = np.sqrt(3) * side_upper_m + point_error_m * np.abs(frame).sum(1)
    lower = projected.max(0) - radius
    upper = projected.min(0) + radius
    if np.any(lower > upper):
        raise ValueError("directional observations contradict diameter/error bound")
    low, high = np.array(baseline["aabb_min"]), np.array(baseline["aabb_max"])
    a, b = np.r_[frame, -frame], np.r_[upper, -lower]
    # Interval propagation derives necessary bounds from each half-space. It
    # need not converge to the tightest box to stay an outer bound. Fixed budget.
    for _ in range(12):
        for normal, offset in zip(a, b, strict=True):
            for j, coefficient in enumerate(normal):
                if abs(coefficient) < 1e-10:
                    continue
                others = [k for k in range(3) if k != j]
                minimum = sum(normal[k] * (low[k] if normal[k] >= 0 else high[k]) for k in others)
                bound = (offset - minimum) / coefficient
                if coefficient > 0:
                    high[j] = min(high[j], bound + 1e-9)
                else:
                    low[j] = max(low[j], bound - 1e-9)
        if np.any(low > high):
            raise ValueError("empty directional intersection")
    return {
        "aabb_min": low.tolist(),
        "aabb_max": high.tolist(),
        "directions": frame.tolist(),
        "projection_min": lower.tolist(),
        "projection_max": upper.tolist(),
        "baseline": baseline,
        "aabb_volume_reduction_fraction": float(
            1 - np.prod(high - low) / np.prod(np.array(baseline["aabb_max"]) - baseline["aabb_min"])
        ),
        "clearance_certified": False,
        "assumptions": [
            "same cube and coordinate-wise error budget as baseline",
            "directions are not assumed to be object axes",
        ],
    }
