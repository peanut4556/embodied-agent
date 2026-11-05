"""Experimental color-reference RGB-D localization, without expected positions."""

from collections import deque

import numpy as np


def reference_components(mask):
    """Four-connected pixel sets; never choose by expected world position."""
    remaining = set(zip(*np.nonzero(mask)))
    components = []
    while remaining:
        seed = remaining.pop()
        pending, pixels = deque([seed]), [seed]
        while pending:
            y, x = pending.popleft()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
                    pixels.append(neighbor)
        components.append(np.array(pixels))
    return components


def locate_references(
    rgb,
    depth,
    camera_position,
    camera_rotation,
    fovy,
    *,
    isolate_components=False,
    diagnostics=None,
    min_depth_fraction=None,
):
    if min_depth_fraction is not None and (
        not isolate_components
        or not np.isfinite(min_depth_fraction)
        or not 0 < min_depth_fraction <= 1
    ):
        raise ValueError("depth coverage requires isolated components and a fraction in (0, 1]")
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
        valid = mask & np.isfinite(depth) & (depth > 0)
        y, x = np.nonzero(valid)
        if isolate_components:
            components = reference_components(mask if min_depth_fraction is not None else valid)
            supported = [pixels for pixels in components if len(pixels) >= 20]
            if diagnostics is not None:
                diagnostics[key] = {
                    "component_sizes": sorted((len(pixels) for pixels in components), reverse=True),
                    "supported_candidates": len(supported),
                }
            # Even a clipped second candidate makes identity ambiguous. Do not
            # silently prefer the largest or nearest-to-catalog component.
            if len(supported) != 1:
                continue
            y, x = supported[0].T
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
        if min_depth_fraction is not None:
            supported_depth = valid[y, x]
            fraction = float(supported_depth.mean())
            if diagnostics is not None:
                diagnostics[key]["depth_fraction"] = fraction
            if fraction < min_depth_fraction or int(supported_depth.sum()) < 20:
                continue
            y, x = y[supported_depth], x[supported_depth]
        rays = np.column_stack(
            ((x + 0.5 - width / 2) / focal, -(y + 0.5 - height / 2) / focal, -np.ones(len(x)))
        )
        points = (rays * depth[y, x, None]) @ np.asarray(camera_rotation).reshape(3, 3).T
        points += np.asarray(camera_position)
        position = np.median(points, axis=0)
        if np.isfinite(position).all():
            result[key] = position.tolist()
    return result
