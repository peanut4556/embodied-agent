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
