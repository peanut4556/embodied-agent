"""Experimental numerical-stationarity stop; original reference_fit is unchanged."""

import numpy as np

from .reference_fit import SCALES, jacobian, predict

LIMITS = np.array([0.05] * 3 + [np.deg2rad(5)] * 3 + [0.05])
EPS = np.finfo(float).eps
USABLE = ("converged", "numerically_stationary")


def stationary_stop(j, residual, step, parameters, rank):
    """Fixed machine-scale guard, not a certificate of calibration correctness."""
    sse = float(residual @ residual)
    denominator = np.linalg.norm(j, axis=0) * np.linalg.norm(residual)
    cosine = np.divide(
        abs(j.T @ residual), denominator, out=np.zeros(j.shape[1]), where=denominator > 0
    )
    change = j @ step
    gain = float(2 * residual @ change - change @ change)
    relative_gain = gain / sse if sse else None
    interior = bool(
        np.all(abs(parameters) < LIMITS) and np.all(abs(parameters + SCALES * step) < LIMITS)
    )
    stop = bool(
        rank == 7
        and sse > 0
        and interior
        and np.max(cosine) <= np.sqrt(EPS)
        and 0 <= gain <= EPS * sse
    )
    return {
        "eligible": stop,
        "rank": int(rank),
        "interior_full_step": interior,
        "max_residual_column_cosine": float(np.max(cosine)),
        "relative_linear_gain": relative_gain,
        "sse_m2": sse,
    }


def fit_reference_candidate(rays, planes, observed):
    observed = np.asarray(observed, dtype=float)
    p = np.zeros(7)
    if observed.shape != predict(rays, planes, p).shape or not np.isfinite(observed).all():
        raise ValueError("finite matching observations required")
    status = "iteration_limit"
    for iteration in range(20):
        residual = (observed - predict(rays, planes, p)).ravel()
        j = jacobian(rays, planes, p)
        step, _, rank, _ = np.linalg.lstsq(j, residual, rcond=1e-7)
        if rank < 7:
            status = "unidentifiable"
            break
        if np.linalg.norm(step) < 1e-7:
            status = "converged"
            break
        guard = stationary_stop(j, residual, step, p, rank)
        if guard["eligible"]:
            status = "numerically_stationary"
            break
        accepted = False
        for backtrack in range(12):
            trial = p + SCALES * step / 2**backtrack
            if np.any(abs(trial) > LIMITS):
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
    j = jacobian(rays, planes, p)
    residual = (observed - predict(rays, planes, p)).ravel()
    step, _, rank, singular = np.linalg.lstsq(j, residual, rcond=1e-7)
    return {
        "status": status,
        "iterations": iteration + 1,
        "parameters": p.tolist(),
        "rank": int(rank),
        "scaled_condition_number": float(singular[0] / singular[-1]) if rank == 7 else None,
        "residual_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "terminal_guard": stationary_stop(j, residual, step, p, rank),
        "calibration_certified": False,
    }
