"""Conservative palm/finger swept AABBs for linear joint interpolation only."""

import numpy as np


def box_aabb(center, rotation, half_size):
    radius = np.abs(np.asarray(rotation).reshape(3, 3)) @ np.asarray(half_size)
    return np.asarray(center) - radius, np.asarray(center) + radius


def overlaps(a, b, margin=0.0):
    return bool(
        np.all(np.asarray(a[0]) <= np.asarray(b[1]) + margin)
        and np.all(np.asarray(b[0]) <= np.asarray(a[1]) + margin)
    )


def swept_box(start_box, end_box, q0, q1, lever_radii):
    q0, q1 = np.asarray(q0), np.asarray(q1)
    radii = np.asarray(lever_radii)
    if (
        q0.shape != (5,)
        or q1.shape != (5,)
        or radii.shape != (3,)
        or not np.isfinite(np.r_[q0, q1, radii]).all()
        or np.any(radii < 0)
    ):
        raise ValueError("invalid joint sweep inputs")
    # Any point is within half the total path-length bound of its nearer endpoint.
    travel = float(radii @ np.abs(q1[:3] - q0[:3]) + np.abs(q1[3:] - q0[3:]).sum())
    return np.minimum(start_box[0], end_box[0]) - travel / 2, np.maximum(
        start_box[1], end_box[1]
    ) + travel / 2


def refine_overlap(box_at, q0, q1, lever_radii, obstacle, max_depth=8):
    """Classify one coarse interval, retaining uncertainty at the depth limit.

    box_at takes a joint vector and returns a conservative static AABB. A static
    witness only proves envelope overlap, never physical collision. Clear means
    every leaf interval has a disjoint swept bound, conditional on input bounds.
    """
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or not 0 <= max_depth <= 12:
        raise ValueError("max_depth must be an integer between 0 and 12")
    q0, q1 = np.asarray(q0), np.asarray(q1)
    cache = {}
    visited = 0
    deepest = 0
    unresolved = []
    witnesses = []

    def pose(t):
        if t not in cache:
            cache[t] = box_at(q0 + t * (q1 - q0))
        return cache[t]

    def visit(a, b, depth):
        nonlocal visited, deepest
        visited += 1
        deepest = max(deepest, depth)
        qa, qb = q0 + a * (q1 - q0), q0 + b * (q1 - q0)
        sweep = swept_box(pose(a), pose(b), qa, qb, lever_radii)
        if not overlaps(sweep, obstacle):
            return "clear"
        for t in (a, b):
            if overlaps(pose(t), obstacle):
                witnesses.append(t)
                return "static_envelope_overlap"
        if depth == max_depth:
            unresolved.append([a, b])
            return "unresolved"
        mid = (a + b) / 2
        left = visit(a, mid, depth + 1)
        right = visit(mid, b, depth + 1)
        if "static_envelope_overlap" in (left, right):
            return "static_envelope_overlap"
        if "unresolved" in (left, right):
            return "unresolved"
        return "clear"

    status = visit(0.0, 1.0, 0)
    return {
        "status": status,
        "nodes_visited": visited,
        "deepest_level": deepest,
        "fk_evaluations": len(cache),
        "static_witness_fractions": sorted(set(witnesses)),
        "unresolved_intervals": unresolved,
    }
