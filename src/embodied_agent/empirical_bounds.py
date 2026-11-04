"""Finite-sample position budgets, not certified error bounds."""

import math

from .reference_confirmation import point3


def estimate_profile_bounds(samples, *, padding_m=0.0005, min_pairs=6, min_pair_fraction=1.0):
    """Fit per-axis maxima from complete paired calibration observations only.

    Samples contain truth_xyz, primary and secondary in corrected world axes.
    Missing pairs are counted; the requested coverage floor must be met.
    Padding is a fixed engineering margin, not a statistical confidence level.
    The caller must freeze these results before collecting validation samples.
    """
    if type(min_pairs) is not int or min_pairs < 1:
        raise ValueError("positive minimum pair count required")
    fraction = float(min_pair_fraction)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("pair coverage fraction must be in (0, 1]")
    padding = float(padding_m)
    if not math.isfinite(padding) or padding < 0:
        raise ValueError("finite nonnegative padding required")
    rows = list(samples)
    if len(rows) < min_pairs:
        raise ValueError("insufficient calibration pairs")
    maxima = [0.0] * 3
    missing = []
    for index, row in enumerate(rows):
        truth = point3(row["truth_xyz"])
        if row["primary"] is None or row["secondary"] is None:
            missing.append(index)
            continue
        for view in ("primary", "secondary"):
            point = point3(row[view])
            for axis in range(3):
                error = abs(point[axis] - truth[axis])
                if not math.isfinite(error):
                    raise ValueError("nonfinite calibration error")
                maxima[axis] = max(maxima[axis], error)
    complete = len(rows) - len(missing)
    if complete < min_pairs or complete / len(rows) < fraction:
        raise ValueError("insufficient complete calibration pair coverage")
    bounds = [value + padding for value in maxima]
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("nonfinite padded budget")
    return {
        "method": "paired_axis_max_plus_fixed_padding_v1",
        "calibration_pairs": len(rows),
        "complete_pairs": complete,
        "missing_pair_indices": missing,
        "pair_coverage": complete / len(rows),
        "min_pair_fraction": fraction,
        "max_observed_error_m": maxima,
        "padding_m": padding,
        "position_error_m": bounds,
        "certified": False,
    }
