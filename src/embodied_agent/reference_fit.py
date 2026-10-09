"""Bounded offline Gauss-Newton fit of a known-plane camera/depth model."""

import numpy as np

from .multiplane_reference import plane_depth, rotation_vector

SCALES = np.array([0.001] * 3 + [np.deg2rad(0.1)] * 3 + [0.001])


def predict(rays, planes, parameters):
    p = np.asarray(parameters, dtype=float)
    if p.shape != (7,) or not np.isfinite(p).all():
        raise ValueError("seven finite parameters required")
    return np.array(
        [
            plane_depth(rays, np.array([0.4, 0, 1.2]) + p[:3], rotation_vector(p[3:6]), n, d) + p[6]
            for n, d in planes
        ]
    )


def jacobian(rays, planes, parameters):
    columns = []
    for i in range(7):
        delta = np.zeros(7)
        delta[i] = 1e-6
        columns.append(
            (
                (
                    predict(rays, planes, parameters + delta)
                    - predict(rays, planes, parameters - delta)
                )
                / (2e-6)
            ).ravel()
            * SCALES[i]
        )
    return np.array(columns).T


def fit_reference(rays, planes, observed):
    observed = np.asarray(observed, dtype=float)
    p = np.zeros(7)
    if observed.shape != predict(rays, planes, p).shape or not np.isfinite(observed).all():
        raise ValueError("finite matching observations required")
    limits = np.array([0.05] * 3 + [np.deg2rad(5)] * 3 + [0.05])
    status = "iteration_limit"
    for iteration in range(20):
        predicted = predict(rays, planes, p)
        residual = (observed - predicted).ravel()
        j = jacobian(rays, planes, p)
        step, _, rank, _ = np.linalg.lstsq(j, residual, rcond=1e-7)
        if rank < 7:
            status = "unidentifiable"
            break
        if np.linalg.norm(step) < 1e-7:
            status = "converged"
            break
        accepted = False
        for backtrack in range(12):
            trial = p + SCALES * step / (2**backtrack)
            if np.any(np.abs(trial) > limits):
                continue
            try:
                trial_residual = (observed - predict(rays, planes, trial)).ravel()
            except ValueError:
                continue
            if np.dot(trial_residual, trial_residual) <= np.dot(residual, residual):
                p = trial
                accepted = True
                break
        if not accepted:
            status = "stalled"
            break
    singular = np.linalg.svd(jacobian(rays, planes, p), compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-7))
    return {
        "status": status,
        "iterations": iteration + 1,
        "parameters": p.tolist(),
        "residual_rmse_m": float(np.sqrt(np.mean((observed - predict(rays, planes, p)) ** 2))),
        "rank": rank,
        "minimum_singular_value_per_sqrt_sample": float(singular[-1] / np.sqrt(observed.size)),
        "scaled_condition_number": float(singular[0] / singular[-1]) if rank == 7 else None,
        "calibration_certified": False,
    }
