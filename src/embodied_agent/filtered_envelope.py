"""Observation-only integration; missing evidence remains unknown, never free space."""

import numpy as np

from .conservative_geometry import envelope
from .rgbd_filter import filter_masks, wall_components
from .visible_geometry import point_cloud


def estimate(rgb, depth, position, rotation, fovy):
    points, valid = point_cloud(rgb, depth, position, rotation, fovy)
    masks = filter_masks(rgb, depth)
    reliable = points[masks["combined"]]
    if len(reliable) < 30:
        return {
            "status": "unknown",
            "reason": "insufficient reliable object points",
            "reliable_points": len(reliable),
            "clearance_certified": False,
        }
    raw = points[masks["raw"]]
    r, g, b = np.asarray(rgb, dtype=float).transpose(2, 0, 1)
    wall = (
        valid
        & (g > 70)
        & (g > 1.3 * r)
        & (b > 1.2 * r)
        & (points[:, :, 0] > raw[:, 0].max())
        & (points[:, :, 1] >= raw[:, 1].min() - 0.005)
        & (points[:, :, 1] <= raw[:, 1].max() + 0.005)
        & (points[:, :, 2] > 0.04)
    )
    groups = wall_components(wall, points)
    wall_x = groups[0]["min_x"] if groups else None
    bound = envelope(reliable, 0.05, 0.003, wall_x, 0.003 if groups else None)
    return {
        "status": "conditional_bound" if groups else "object_only",
        "reliable_points": len(reliable),
        "raw_points": len(raw),
        "wall_components": len(groups),
        "visible_filtered_x_gap_m": None
        if wall_x is None
        else wall_x - float(reliable[:, 0].max()),
        "bound": bound,
        "clearance_certified": False,
    }
