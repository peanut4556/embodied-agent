"""Fixed synthetic depth slopes for auditing shared versus one-sided bias."""

import numpy as np


def depth_ramp(depth, red_mask, amplitude_m, mode):
    depth, red = np.asarray(depth), np.asarray(red_mask, dtype=bool)
    if (
        depth.ndim != 2
        or red.shape != depth.shape
        or not np.isfinite(amplitude_m)
        or mode not in ("none", "shared", "candidate", "validation")
    ):
        raise ValueError("invalid ramp inputs")
    if not red.any():
        return depth.copy(), np.zeros(depth.shape)
    y, x = np.indices(depth.shape)
    xx = x[red]
    half_width = (xx.max() - xx.min()) / 2
    if half_width == 0:
        raise ValueError("red support needs horizontal extent")
    ramp = amplitude_m * np.clip((x - (xx.min() + xx.max()) / 2) / half_width, -1, 1)
    candidate = ((y // 8 + x // 8) % 2) == 0
    apply = red.copy()
    if mode == "candidate":
        apply &= candidate
    elif mode == "validation":
        apply &= ~candidate
    elif mode == "none":
        apply[:] = False
    delta = np.where(apply, ramp, 0.0)
    return depth + delta, delta
