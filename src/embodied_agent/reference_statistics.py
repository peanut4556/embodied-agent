"""First-order heldout mean test with explicit noise correlations."""

from statistics import NormalDist

import numpy as np

from .reference_fit import jacobian


def mean_prediction_uncertainty(
    rays, planes, parameters, sigma_pixel_m, sigma_exposure_m=0.0, sigma_global_m=0.0
):
    sigmas = np.array([sigma_pixel_m, sigma_exposure_m, sigma_global_m], dtype=float)
    if len(planes) != 4 or not np.isfinite(sigmas).all() or np.any(sigmas < 0):
        raise ValueError("four planes and finite nonnegative noise scales required")
    p = np.asarray(parameters, dtype=float)
    train = jacobian(rays, planes[:3], p)
    validation = jacobian(rays, planes[3:], p)
    u, s, v = np.linalg.svd(train, full_matrices=False)
    if np.sum(s > s[0] * 1e-7) != 7:
        raise ValueError("training Jacobian not identifiable")
    weights = validation.mean(0) @ ((v.T / s) @ u.T)
    sums = weights.reshape(3, len(rays)).sum(1)
    variances = {
        "independent_pixels": float(sigma_pixel_m**2 * (1 / len(rays) + weights @ weights)),
        "shared_within_exposure": float(sigma_exposure_m**2 * (1 + sums @ sums)),
        "shared_across_all_exposures": float(sigma_global_m**2 * (1 - weights.sum()) ** 2),
    }
    standard_error = float(np.sqrt(sum(variances.values())))
    # A numerical floor avoids division by zero, not a calibrated uncertainty.
    threshold = max(1e-9, NormalDist().inv_cdf(0.995) * standard_error)
    return {
        "variance_components_m2": variances,
        "standard_error_m": standard_error,
        "two_sided_99_percent_threshold_m": threshold,
        "calibration_certified": False,
    }


def wilson_interval(alarms, total):
    if not 0 <= alarms <= total:
        raise ValueError("invalid counts")
    if total == 0:
        return None
    z = NormalDist().inv_cdf(0.975)
    p = alarms / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(max(0, center - radius)), float(min(1, center + radius))]
