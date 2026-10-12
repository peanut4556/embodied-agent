"""Conservative Gaussian variance bounds from independent repeated exposures.

Laurent--Massart chi-square tails, followed by a five-event union bound.
Only calibration observations enter this construction. Nonlinear fitted prediction
weights still make the resulting diagnostic a first-order approximation.
"""

import numpy as np

from .repeated_exposure import estimate_repeated_noise


def variance_bounds(estimate, degrees_of_freedom, failure_probability):
    """Each returned endpoint has one-sided failure probability at most p."""
    if (
        not np.isfinite([estimate, degrees_of_freedom, failure_probability]).all()
        or estimate < 0
        or degrees_of_freedom <= 0
        or not 0 < failure_probability < 1
    ):
        raise ValueError(
            "finite nonnegative variance, positive df and probability in (0, 1) required"
        )
    t = -np.log(failure_probability)
    df = float(degrees_of_freedom)
    lower_chi = df - 2 * np.sqrt(df * t)
    if lower_chi <= 0:
        raise ValueError("insufficient repeats for finite concentration upper bound")
    return {
        "lower": float(df * estimate / (df + 2 * np.sqrt(df * t) + 2 * t)),
        "upper": float(df * estimate / lower_chi),
    }


def bound_repeated_noise(observations, calibration_failure_probability=0.005):
    estimate = estimate_repeated_noise(observations)
    if (
        not np.isfinite(calibration_failure_probability)
        or not 0 < calibration_failure_probability < 1
    ):
        raise ValueError("calibration failure probability must be in (0, 1)")
    x = np.asarray(observations, dtype=float)
    repeats, exposures, pixels = x.shape
    # Orthogonal exposure contrasts cancel the global component and static means.
    means = x.mean(axis=2)
    centered = means - means.mean(axis=0, keepdims=True)
    contrasts = centered - centered.mean(axis=1, keepdims=True)
    a = estimate["raw_variance_components_m2"]["pixel"]
    b = float(np.sum(contrasts**2) / ((repeats - 1) * (exposures - 1)))
    c = float(np.var(means.mean(axis=1), ddof=1))
    # A = pixel variance; B = exposure variance + A/P;
    # C = global variance + B/E. Their scaled estimates are chi-squared.
    dfs = [(repeats - 1) * exposures * (pixels - 1), (repeats - 1) * (exposures - 1), repeats - 1]
    bounds = [
        variance_bounds(value, df, calibration_failure_probability / 5)
        for value, df in zip((a, b, c), dfs)
    ]
    upper = [
        bounds[0]["upper"],
        max(0.0, bounds[1]["upper"] - bounds[0]["lower"] / pixels),
        max(0.0, bounds[2]["upper"] - bounds[1]["lower"] / exposures),
    ]
    return {
        "estimate": estimate,
        "orthogonal_variance_estimates_m2": [a, b, c],
        "degrees_of_freedom": dfs,
        "orthogonal_variance_bounds_m2": bounds,
        "upper_variance_components_m2": upper,
        "upper_sigmas_m": np.sqrt(upper).tolist(),
        "calibration_failure_probability": calibration_failure_probability,
        "union_bound_events": ["A_lower", "A_upper", "B_lower", "B_upper", "C_upper"],
        "calibration_certified": False,
    }
