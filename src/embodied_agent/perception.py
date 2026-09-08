"""Calibrated overhead RGB-D localization of one upright 5 cm red cube.

Only images and camera calibration enter this module, never simulator object poses.
"""

from collections import deque

import numpy as np


def locate_red_cube(rgb, depth, camera_position, camera_rotation, fovy):
    rgb = np.asarray(rgb, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if rgb.shape != (*depth.shape, 3) or depth.ndim != 2:
        raise ValueError("RGB and depth dimensions do not match")
    red, green, blue = rgb.transpose(2, 0, 1)
    mask = (red > 70) & (red > 1.6 * green) & (red > 1.4 * blue)
    mask &= np.isfinite(depth) & (depth > 0)
    # Separate candidates instead of averaging multiple red objects together.
    remaining = set(zip(*np.nonzero(mask)))
    components = []
    while remaining:
        seed = remaining.pop()
        queue, pixels = deque([seed]), [seed]
        while queue:
            y, x = queue.popleft()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    pixels.append(neighbor)
        if len(pixels) >= 20:
            components.append(pixels)
    if len(components) != 1:
        raise ValueError(f"vision requires one visible red cube; found {len(components)}")
    pixels = np.array(components[0])
    y, x = pixels.T
    height, width = depth.shape
    if x.min() == 0 or y.min() == 0 or x.max() == width - 1 or y.max() == height - 1:
        raise ValueError("vision target is clipped by image boundary")
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    z = depth[y, x]
    rays = np.column_stack(
        ((x + 0.5 - width / 2) / focal, -(y + 0.5 - height / 2) / focal, -np.ones(len(x)))
    )
    points = (rays * z[:, None]) @ np.asarray(camera_rotation).reshape(3, 3).T
    points += np.asarray(camera_position)
    # Overhead view: select the top face, discard side faces and edge artifacts.
    top_z = np.percentile(points[:, 2], 90)
    top = points[np.abs(points[:, 2] - top_z) < 0.003]
    if len(top) < 20:
        raise ValueError("vision target has insufficient depth support")
    extent = np.ptp(top[:, :2], axis=0)
    if np.any(extent < 0.038) or np.any(extent > 0.060):
        raise ValueError("vision target size inconsistent or substantially occluded")
    center = (top.min(axis=0) + top.max(axis=0)) / 2
    center[2] = np.median(top[:, 2]) - 0.025
    return {
        "source": "rgbd",
        "xyz": center.tolist(),
        "pixels": len(top),
        "bbox": [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
    }
