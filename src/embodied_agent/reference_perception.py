"""Experimental color-reference RGB-D localization, without expected positions."""

import numpy as np


def locate_references(rgb, depth, camera_position, camera_rotation, fovy):
    rgb, depth = np.asarray(rgb, dtype=float), np.asarray(depth, dtype=float)
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError("matching RGB and depth arrays required")
    red, green, blue = rgb.transpose(2, 0, 1)
    masks = {
        "blue": (blue > 70) & (blue > 1.6 * red) & (blue > 1.6 * green),
        "yellow": (red > 70) & (green > 70) & (red > 1.6 * blue) & (green > 1.6 * blue),
        "magenta": (red > 70) & (blue > 70) & (red > 1.6 * green) & (blue > 1.6 * green),
    }
    height, width = depth.shape
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    result = {}
    for key, mask in masks.items():
        y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
        if len(x) < 20:
            continue
        # Reject clipped or spread-out color regions rather than merging them.
        if (
            x.min() == 0
            or y.min() == 0
            or x.max() == width - 1
            or y.max() == height - 1
            or np.ptp(x) > 48
            or np.ptp(y) > 48
        ):
            continue
        rays = np.column_stack(
            ((x + 0.5 - width / 2) / focal, -(y + 0.5 - height / 2) / focal, -np.ones(len(x)))
        )
        points = (rays * depth[y, x, None]) @ np.asarray(camera_rotation).reshape(3, 3).T
        points += np.asarray(camera_position)
        position = np.median(points, axis=0)
        if np.isfinite(position).all():
            result[key] = position.tolist()
    return result
