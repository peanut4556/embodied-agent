"""RGB-D visible-surface diagnostics, never a full-shape or free-space certificate."""

import numpy as np


def point_cloud(rgb, depth, position, rotation, fovy):
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError("RGB-D dimensions differ")
    height, width = depth.shape
    y, x = np.indices(depth.shape)
    focal = height / (2 * np.tan(np.deg2rad(float(fovy)) / 2))
    rays = np.stack(
        ((x + 0.5 - width / 2) / focal, -(y + 0.5 - height / 2) / focal, -np.ones_like(depth)),
        axis=-1,
    )
    points = (rays * depth[:, :, None]) @ np.asarray(rotation).reshape(3, 3).T + np.asarray(
        position
    )
    return points, np.isfinite(depth) & (depth > 0)


def dominant_plane(points):
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 30
        or not np.isfinite(points).all()
    ):
        raise ValueError("at least 30 finite surface points required")
    rng = np.random.default_rng(73)
    best = None
    for _ in range(256):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        keep = np.abs((points - a) @ normal) < 0.0015
        if best is None or keep.sum() > best.sum():
            best = keep
    if best is None or best.sum() < 30:
        raise ValueError("insufficient plane support")
    cloud = points[best]
    _, _, vectors = np.linalg.svd(cloud - cloud.mean(0), full_matrices=False)
    normal = vectors[-1]
    if normal[2] < 0:
        normal = -normal
    return {
        "normal": normal.tolist(),
        "tilt_degrees": float(np.rad2deg(np.arccos(np.clip(normal[2], 0, 1)))),
        "inlier_count": int(best.sum()),
        "inlier_fraction": float(best.mean()),
        "residual_rmse_m": float(np.sqrt(np.mean(((cloud - cloud.mean(0)) @ normal) ** 2))),
    }


def estimate(rgb, depth, position, rotation, fovy):
    points, valid = point_cloud(rgb, depth, position, rotation, fovy)
    r, g, b = np.asarray(rgb, dtype=float).transpose(2, 0, 1)
    red = valid & (r > 70) & (r > 1.6 * g) & (r > 1.4 * b)
    object_points = points[red]
    plane = dominant_plane(object_points)
    low, high = object_points.min(0), object_points.max(0)
    # Fixture-specific teal obstacle to the right, overlapping the visible object's Y span.
    obstacle = valid & (g > 70) & (g > 1.3 * r) & (b > 1.2 * r)
    obstacle &= (
        (points[:, :, 0] > high[0])
        & (points[:, :, 1] >= low[1] - 0.005)
        & (points[:, :, 1] <= high[1] + 0.005)
        & (points[:, :, 2] > 0.04)
    )
    wall = points[obstacle]
    wall_x = float(wall[:, 0].min()) if len(wall) >= 20 else None
    return {
        "visible_points": len(object_points),
        "visible_aabb_min": low.tolist(),
        "visible_aabb_max": high.tolist(),
        "dominant_visible_plane": plane,
        "obstacle_points": len(wall),
        "visible_wall_x": wall_x,
        "visible_x_gap_m": None if wall_x is None else wall_x - float(high[0]),
        "clearance_certified": False,
        "scope": "visible surfaces only; hidden geometry and contact gap unresolved",
    }
