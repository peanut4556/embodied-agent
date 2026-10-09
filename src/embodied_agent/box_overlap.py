"""Static oriented-box separation for offline geometry attribution."""

import numpy as np


def obb_overlap(a, b, tolerance=1e-9):
    """Boxes are (center, rotation with column axes, half sizes); touching overlaps.

    The 15 separating-axis test is exact for boxes except the conservative
    numerical tolerance. This is not a swept-volume or physical safety check.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid tolerance")
    boxes = []
    for center, rotation, size in (a, b):
        c, r, s = np.asarray(center), np.asarray(rotation), np.asarray(size)
        if (
            c.shape != (3,)
            or r.shape != (3, 3)
            or s.shape != (3,)
            or not all(np.isfinite(v).all() for v in (c, r, s))
            or np.any(s < 0)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-10, rtol=0)
        ):
            raise ValueError("invalid oriented box")
        boxes.append((c, r, s))
    ca, ra, sa = boxes[0]
    cb, rb, sb = boxes[1]
    axes = [*ra.T, *rb.T, *(np.cross(u, v) for u in ra.T for v in rb.T)]
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-12:
            continue
        axis = axis / length
        radius = np.abs(axis @ ra) @ sa + np.abs(axis @ rb) @ sb
        if abs(axis @ (cb - ca)) > radius + tolerance:
            return False
    return True


def aabb_as_obb(bounds):
    low, high = np.asarray(bounds[0]), np.asarray(bounds[1])
    return (low + high) / 2, np.eye(3), (high - low) / 2
