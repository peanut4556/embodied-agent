"""Offline first-order translation uncertainty, not a model-validity certificate."""

from statistics import NormalDist

import numpy as np

from .reference_fit import SCALES, jacobian


def translation_covariance(rays, planes, parameters, sigmas):
    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.shape != (3,) or not np.isfinite(sigmas).all() or np.any(sigmas < 0):
        raise ValueError("three finite nonnegative noise scales required")
    if len(planes) != 3:
        raise ValueError("three training planes required")
    j = jacobian(rays, planes, parameters)
    u, s, v = np.linalg.svd(j, full_matrices=False)
    if np.sum(s > s[0] * 1e-7) != 7:
        raise ValueError("training geometry is unidentifiable")
    # Convert the scaled least-squares inverse to physical translation units.
    weights = SCALES[:3, None] * ((v.T / s) @ u.T)[:3]
    exposure = weights.reshape(3, 3, len(rays)).sum(axis=2)
    shared = exposure.sum(axis=1)
    return (
        sigmas[0] ** 2 * (weights @ weights.T)
        + sigmas[1] ** 2 * (exposure @ exposure.T)
        + sigmas[2] ** 2 * np.outer(shared, shared)
    )


def translation_gate(covariance, error_limit_m=0.005, failure_probability=0.045):
    covariance = np.asarray(covariance, dtype=float)
    if (
        covariance.shape != (3, 3)
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-18)
        or np.min(np.linalg.eigvalsh(covariance)) < -1e-18
        or not np.isfinite([error_limit_m, failure_probability]).all()
        or error_limit_m <= 0
        or not 0 < failure_probability < 1
    ):
        raise ValueError("finite positive semidefinite covariance, limit and probability required")
    # Union bound over three two-sided coordinate intervals; no independence assumed.
    z = NormalDist().inv_cdf(1 - failure_probability / 6)
    radius = float(z * np.sqrt(max(0.0, float(np.trace(covariance)))))
    return {
        "radius_m": radius,
        "passes": bool(radius <= error_limit_m),
        "calibration_certified": False,
    }
