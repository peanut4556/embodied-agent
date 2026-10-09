"""Necessary size checks and diagnostic cube-face relations, never pose certification."""

from itertools import combinations

import numpy as np


def diameter(points):
    maximum = 0.0
    for start in range(0, len(points), 128):
        delta = points[start : start + 128, None] - points[None]
        maximum = max(maximum, float(np.sqrt(np.max(np.sum(delta**2, axis=2)))))
    return maximum


def cube_consistency(points, groups, side_upper_m=0.05, point_error_m=0.003):
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or not np.isfinite([side_upper_m, point_error_m]).all()
        or side_upper_m <= 0
        or point_error_m < 0
    ):
        raise ValueError("invalid points or geometry budget")
    if len(points) < 30:
        return {"status": "unknown", "reason": "insufficient_points", "clearance_certified": False}
    error = 2 * np.sqrt(3) * point_error_m
    limit = np.sqrt(3) * side_upper_m + error
    extent = diameter(points)
    faces = []
    normals = []
    for g in groups:
        ids = np.asarray(g["indices"])
        n = np.asarray(g["normal"], dtype=float)
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or len(ids) == 0
            or np.any(ids < 0)
            or np.any(ids >= len(points))
            or n.shape != (3,)
            or not np.isfinite(n).all()
            or not np.isclose(np.linalg.norm(n), 1)
        ):
            raise ValueError("invalid group")
        d = diameter(points[ids])
        faces.append(
            {
                "points": len(ids),
                "diameter_m": d,
                "conditional_same_face_limit_m": np.sqrt(2) * side_upper_m + error,
                "same_face_size_alarm": bool(d > np.sqrt(2) * side_upper_m + error + 1e-9),
            }
        )
        normals.append(n)
    pairs = []
    for i, j in combinations(range(len(normals)), 2):
        angle = float(np.rad2deg(np.arccos(np.clip(abs(normals[i] @ normals[j]), 0, 1))))
        deviation = min(angle, 90 - angle)
        pairs.append(
            {
                "groups": [i, j],
                "acute_angle_degrees": angle,
                "deviation_from_parallel_or_orthogonal_degrees": deviation,
                "relation_alarm": bool(deviation > 10),
            }
        )
    return {
        "status": "evaluated",
        "object_diameter_m": extent,
        "object_diameter_limit_m": limit,
        "size_contradiction": bool(extent > limit + 1e-9),
        "faces": faces,
        "pairs": pairs,
        "face_size_alarm": any(f["same_face_size_alarm"] for f in faces),
        "relation_alarm": any(p["relation_alarm"] for p in pairs),
        "multiple_face_evidence": "available" if pairs else "insufficient",
        "angle_bound_certified": False,
        "clearance_certified": False,
    }
