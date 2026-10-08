"""Analytic box-ray and surface distances used only for simulation scoring."""

import numpy as np


def ray_box_depth(origin, directions, center, rotation, half_size):
    origin = (np.asarray(origin) - center) @ rotation
    rays = np.asarray(directions) @ rotation
    size = np.asarray(half_size)
    parallel = np.abs(rays) < 1e-12
    divisor = np.where(parallel, 1.0, rays)
    a, b = (-size - origin) / divisor, (size - origin) / divisor
    lower = np.where(parallel, -np.inf, np.minimum(a, b))
    upper = np.where(parallel, np.inf, np.maximum(a, b))
    near, far = lower.max(-1), upper.min(-1)
    hit = (far >= np.maximum(near, 0)) & ~np.any(parallel & (np.abs(origin) > size), axis=-1)
    return np.where(hit, np.where(near >= 0, near, far), np.nan)


def surface_distance(points, center, rotation, half_size):
    q = np.abs((np.asarray(points) - center) @ rotation) - half_size
    return np.abs(np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0))


def erode(mask):
    padded = np.pad(mask, 1)
    return mask & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {
        "count": len(values),
        "median_m": float(np.median(values)) if len(values) else None,
        "p95_m": float(np.percentile(values, 95)) if len(values) else None,
        "max_m": float(values.max()) if len(values) else None,
    }
