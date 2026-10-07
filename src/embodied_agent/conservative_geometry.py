"""Conditional outer bounds from a known cube diameter, not a pose reconstruction."""

import numpy as np


def envelope(points, side_upper_m, point_error_m, wall_x=None, wall_error_m=None):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("finite nonempty object points required")
    if (
        not np.isfinite([side_upper_m, point_error_m]).all()
        or side_upper_m <= 0
        or point_error_m < 0
    ):
        raise ValueError("invalid size or error bound")
    diameter = np.sqrt(3) * side_upper_m
    radius = diameter + point_error_m
    # Any point on/in a cube is at most its diameter from every other cube point.
    low = points.max(0) - radius
    high = points.min(0) + radius
    if np.any(low > high):
        raise ValueError("observations contradict the assumed diameter/error bound")
    lower_gap = None
    if wall_x is not None:
        if (
            wall_error_m is None
            or not np.isfinite([wall_x, wall_error_m]).all()
            or wall_error_m < 0
        ):
            raise ValueError("finite wall uncertainty required")
        lower_gap = float(wall_x - wall_error_m - high[0])
    return {
        "aabb_min": low.tolist(),
        "aabb_max": high.tolist(),
        "side_upper_m": side_upper_m,
        "point_error_m": point_error_m,
        "wall_error_m": wall_error_m,
        "conditional_x_gap_lower_m": lower_gap,
        "conditional_positive_gap": bool(lower_gap is not None and lower_gap > 0),
        "clearance_certified": False,
        "assumptions": [
            "all input points lie within the stated coordinate error of this same cube",
            "cube side length does not exceed the stated prior",
            "wall observation is the nearest relevant planar X-facing surface with bounded error; no hidden closer obstacles",
        ],
        "scope": "conditional geometric bound; uncalibrated sensing and swept-volume clearance remain unresolved",
    }
