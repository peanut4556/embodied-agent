"""First-order bounded-noise propagation; not a nonlinear calibration certificate."""

import numpy as np

from .reference_fit import jacobian, predict


def propagated_threshold(rays, planes, parameters, noise_bound_m):
    if not np.isfinite(noise_bound_m) or noise_bound_m < 0 or len(planes) != 4:
        raise ValueError("four planes and nonnegative finite noise bound required")
    p = np.asarray(parameters, dtype=float)
    train = jacobian(rays, planes[:3], p)
    validation = jacobian(rays, planes[3:], p)
    u, s, v = np.linalg.svd(train, full_matrices=False)
    if np.sum(s > s[0] * 1e-7) != 7:
        raise ValueError("training Jacobian not identifiable")
    inverse = (v.T / s) @ u.T
    weights = validation @ inverse
    # Each training depth error is bounded by b. Validation has its own b.
    # Units cancel because both Jacobians use identical parameter scaling.
    gain = np.sum(np.abs(weights), axis=1)
    threshold = np.maximum(0.003, noise_bound_m * (1 + gain))
    return threshold, gain


def check_uncertain_reference(rays, planes, observed, fit, noise_bound_m):
    observed = np.asarray(observed, dtype=float)
    if (
        observed.shape != (4, len(rays))
        or not np.isfinite(observed).all()
        or not np.isfinite(noise_bound_m)
        or noise_bound_m < 0
    ):
        raise ValueError("invalid observation or noise budget")
    if fit["status"] != "converged":
        return {
            "status": "unknown",
            "reason": "training_fit_not_converged",
            "calibration_certified": False,
        }
    try:
        threshold, gain = propagated_threshold(rays, planes, fit["parameters"], noise_bound_m)
    except ValueError as exc:
        return {"status": "unknown", "reason": str(exc), "calibration_certified": False}
    residual = observed[3] - predict(rays, planes[3:], fit["parameters"])[0]
    ratio = float(np.quantile(np.abs(residual) / threshold, 0.95))
    return {
        "status": "alarm" if ratio > 1 else "no_alarm",
        "normalized_residual_p95": ratio,
        "threshold_range_m": [float(threshold.min()), float(threshold.max())],
        "noise_gain_range": [float(gain.min()), float(gain.max())],
        "assumed_noise_bound_m": noise_bound_m,
        "calibration_certified": False,
    }
