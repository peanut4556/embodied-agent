"""Experimental translation check against an independently specified position sensor."""

from statistics import NormalDist

import numpy as np


def external_translation_gate(
    translation,
    measurement,
    axes,
    sigma_m,
    error_limit_m=0.005,
    failure_probability=0.01,
    systematic_error_bound_m=0.0,
):
    """Check a measured subspace; only full xyz can admit a 3D translation.

    Measurement and estimate must use the same known coordinate frame and origin.
    The caller supplies a valid sensor noise/bias budget; this function cannot
    discover shared systematic errors or certify that an external sensor is independent.
    """
    p = np.asarray(translation, dtype=float)
    m = np.asarray(measurement, dtype=float)
    axes = tuple(axes)
    if (
        p.shape != (3,)
        or not np.isfinite(p).all()
        or not axes
        or any(type(a) is not int or a not in (0, 1, 2) for a in axes)
        or len(set(axes)) != len(axes)
        or m.shape != (len(axes),)
        or not np.isfinite(m).all()
        or not np.isfinite(
            [sigma_m, error_limit_m, failure_probability, systematic_error_bound_m]
        ).all()
        or sigma_m < 0
        or systematic_error_bound_m < 0
        or error_limit_m <= 0
        or not 0 < failure_probability < 1
    ):
        raise ValueError(
            "finite positions, distinct xyz axes, nonnegative noise/bias and valid budget required"
        )
    z = NormalDist().inv_cdf(1 - failure_probability / (2 * len(axes)))
    sensor_radius = float(z * np.sqrt(len(axes)) * sigma_m + systematic_error_bound_m)
    discrepancy = float(np.linalg.norm(p[list(axes)] - m))
    upper = discrepancy + sensor_radius
    full_rank = len(axes) == 3
    return {
        "observed_dimensions": len(axes),
        "full_translation_observed": full_rank,
        "sensor_radius_m": sensor_radius,
        "discrepancy_m": discrepancy,
        "error_upper_m": upper,
        "projected_passes": bool(upper <= error_limit_m),
        "passes": bool(full_rank and upper <= error_limit_m),
        "calibration_certified": False,
    }
