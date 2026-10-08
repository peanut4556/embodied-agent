"""Observation-only boundary/depth filters and connected wall candidates for auditing."""

from collections import deque

import numpy as np


def erode_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("2D mask required")
    p = np.pad(mask, 1)
    return mask & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def filter_masks(rgb, depth, max_depth_jump=0.005):
    rgb, depth = np.asarray(rgb), np.asarray(depth, dtype=float)
    if (
        depth.ndim != 2
        or rgb.shape != (*depth.shape, 3)
        or not np.isfinite(max_depth_jump)
        or max_depth_jump <= 0
    ):
        raise ValueError("invalid RGB-D/filter settings")
    r, g, b = rgb.astype(float).transpose(2, 0, 1)
    valid = np.isfinite(depth) & (depth > 0)
    red = valid & (r > 70) & (r > 1.6 * g) & (r > 1.4 * b)
    p = np.pad(depth, 1, constant_values=np.nan)
    continuous = valid.copy()
    for neighbor in (p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]):
        continuous &= (
            np.isfinite(neighbor) & (neighbor > 0) & (np.abs(neighbor - depth) <= max_depth_jump)
        )
    edge = erode_mask(red)
    return {
        "raw": red,
        "eroded": edge,
        "continuous": red & continuous,
        "combined": edge & continuous,
    }


def wall_components(mask, points, min_pixels=20):
    mask, points = np.asarray(mask, dtype=bool), np.asarray(points)
    if mask.ndim != 2 or points.shape != (*mask.shape, 3) or min_pixels < 1:
        raise ValueError("invalid wall candidates")
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        queue = deque([first])
        pixels = [first]
        while queue:
            y, x = queue.popleft()
            for item in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if item in remaining:
                    remaining.remove(item)
                    queue.append(item)
                    pixels.append(item)
        if len(pixels) < min_pixels:
            continue
        yy, xx = np.asarray(pixels).T
        cloud = points[yy, xx]
        if not np.isfinite(cloud).all():
            raise ValueError("nonfinite wall points")
        result.append(
            {
                "pixels": pixels,
                "count": len(pixels),
                "min_x": float(cloud[:, 0].min()),
                "max_x": float(cloud[:, 0].max()),
            }
        )
    return sorted(result, key=lambda x: x["min_x"])
