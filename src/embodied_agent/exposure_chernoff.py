"""Invert the optimized chi-square Chernoff tail bound, without tail approximations."""

from functools import lru_cache

import numpy as np

from .exposure_uncertainty import bound_repeated_noise


@lru_cache(maxsize=128)
def chi_square_ratios(df, failure_probability):
    """Return conservative lower/upper X/df endpoints with one-sided error <= p.

    For X~chi-square(df), optimize its MGF to obtain
    P(tail beyond df*r) <= exp(-df*(r-1-log(r))/2).
    Bisection in log(r) avoids cancellation and retains the outer brackets.
    """
    if (
        not np.isfinite([df, failure_probability]).all()
        or df <= 0
        or not 0 < failure_probability < 1
    ):
        raise ValueError("finite positive df and probability in (0, 1) required")
    target = -2 * np.log(failure_probability) / df
    low, high = -target - 2, 0.0
    for _ in range(100):
        mid = (low + high) / 2
        if np.expm1(mid) - mid >= target:
            low = mid
        else:
            high = mid
    lower = float(np.exp(low))
    low, high = 0.0, float(np.log(2 * (target + 2)))
    for _ in range(100):
        mid = (low + high) / 2
        if np.expm1(mid) - mid >= target:
            high = mid
        else:
            low = mid
    upper = float(np.exp(high))
    if lower <= 0 or not np.isfinite(upper):
        raise ValueError("variance bound outside floating-point range")
    return lower, upper


def tighten_noise_bound(calibration_bound):
    """Reuse calibration-only orthogonal sufficient statistics from the LM rule."""
    probability = calibration_bound["calibration_failure_probability"] / 5
    endpoints = []
    for value, df in zip(
        calibration_bound["orthogonal_variance_estimates_m2"],
        calibration_bound["degrees_of_freedom"],
    ):
        low, high = chi_square_ratios(df, probability)
        endpoints.append({"lower": value / high, "upper": value / low})
    _, exposures, pixels = calibration_bound["estimate"]["shape"]
    upper = [
        endpoints[0]["upper"],
        max(0.0, endpoints[1]["upper"] - endpoints[0]["lower"] / pixels),
        max(0.0, endpoints[2]["upper"] - endpoints[1]["lower"] / exposures),
    ]
    return {
        "method": "optimized_chi_square_chernoff",
        "orthogonal_variance_bounds_m2": endpoints,
        "upper_variance_components_m2": upper,
        "upper_sigmas_m": np.sqrt(upper).tolist(),
        "calibration_failure_probability": calibration_bound["calibration_failure_probability"],
        "calibration_certified": False,
    }


def bound_repeated_noise_chernoff(observations, calibration_failure_probability=0.005):
    calibration = bound_repeated_noise(observations, calibration_failure_probability)
    return {"calibration": calibration, "chernoff": tighten_noise_bound(calibration)}
