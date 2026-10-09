"""Deterministic visible-plane proposals, not certified object face associations."""

import numpy as np


def plane_groups(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("finite Nx3 points required")
    remaining = np.arange(len(points))
    groups = []
    rng = np.random.default_rng(73)
    for _ in range(3):
        if len(remaining) < 30:
            break
        cloud = points[remaining]
        best = np.zeros(len(cloud), dtype=bool)
        for _ in range(256):
            a, b, c = cloud[rng.choice(len(cloud), 3, replace=False)]
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            keep = np.abs((cloud - a) @ normal) <= 0.0015
            if keep.sum() > best.sum():
                best = keep
        if best.sum() < 30:
            break
        for _ in range(2):
            center = cloud[best].mean(0)
            _, _, axes = np.linalg.svd(cloud[best] - center, full_matrices=False)
            best = np.abs((cloud - center) @ axes[-1]) <= 0.0015
            if best.sum() < 30:
                break
        if best.sum() < 30:
            break
        selected = cloud[best]
        center = selected.mean(0)
        _, _, axes = np.linalg.svd(selected - center, full_matrices=False)
        normal = axes[-1]
        if normal[np.argmax(np.abs(normal))] < 0:
            normal = -normal
        spans = np.ptp((selected - center) @ axes[:2].T, axis=0)
        residual = np.abs((selected - center) @ normal)
        # Line-like or tiny support does not establish a plane normal.
        accepted = bool(np.min(spans) >= 0.01 and np.max(residual) <= 0.002)
        groups.append(
            {
                "indices": remaining[best].tolist(),
                "normal": normal.tolist(),
                "center": center.tolist(),
                "tangent_spans_m": spans.tolist(),
                "residual_rmse_m": float(np.sqrt(np.mean(residual**2))),
                "residual_max_m": float(residual.max()),
                "proposal_accepted": accepted,
                "reason": "candidate_plane_only" if accepted else "insufficient_span_or_residual",
                "angle_bound_certified": False,
            }
        )
        remaining = remaining[~best]
    return {
        "groups": groups,
        "unassigned_indices": remaining.tolist(),
        "status": "candidate_groups" if any(g["proposal_accepted"] for g in groups) else "unknown",
        "angle_bound_certified": False,
    }
