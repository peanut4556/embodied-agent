"""Known clear tabletop ROI check; fixed calibration is an explicit assumption."""

import numpy as np


def reference_geometry(shape, position, rotation, fovy):
    position, rotation = np.asarray(position), np.asarray(rotation).reshape(3, 3)
    if (
        len(shape) != 2
        or min(shape) <= 0
        or position.shape != (3,)
        or not np.isfinite(position).all()
        or not np.isfinite(rotation).all()
        or not 0 < fovy < 180
    ):
        raise ValueError("invalid camera")
    y, x = np.indices(shape)
    h, w = shape
    focal = h / (2 * np.tan(np.deg2rad(fovy) / 2))
    rays = (
        np.stack(((x + 0.5 - w / 2) / focal, -(y + 0.5 - h / 2) / focal, -np.ones(shape)), axis=-1)
        @ rotation.T
    )
    expected = np.divide(
        -position[2],
        rays[:, :, 2],
        out=np.full(shape, np.nan),
        where=np.abs(rays[:, :, 2]) > 1e-10,
    )
    intersection = position + rays * expected[:, :, None]
    # Known unoccupied fixture patch, chosen by ray/plane intersection ONLY.
    roi = (
        (expected > 0)
        & (intersection[:, :, 0] >= 0.20)
        & (intersection[:, :, 0] <= 0.65)
        & (intersection[:, :, 1] >= 0.20)
        & (intersection[:, :, 1] <= 0.30)
    )
    return expected, roi


def check_reference(depth, position, rotation, fovy):
    depth = np.asarray(depth, dtype=float)
    expected, roi = reference_geometry(depth.shape, position, rotation, fovy)
    usable = roi & np.isfinite(depth) & (depth > 0)
    total, count = int(roi.sum()), int(usable.sum())
    result = {"reference_pixels": total, "valid_pixels": count, "clearance_certified": False}
    if total < 100 or count < 0.8 * total:
        return result | {"status": "unknown", "reason": "insufficient_reference_coverage"}
    residual = depth[usable] - expected[usable]
    p95 = float(np.quantile(np.abs(residual), 0.95))
    return result | {
        "status": "alarm" if p95 > 0.003 else "no_alarm",
        "absolute_residual_p95_m": p95,
        "signed_residual_median_m": float(np.median(residual)),
        "maximum_absolute_residual_m": float(np.max(np.abs(residual))),
        "scope": "known reference ROI only; object-local errors may be invisible",
    }
