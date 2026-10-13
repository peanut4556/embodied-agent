"""Offline instrumented replay of reference_fit; no change to production stopping rules."""

import numpy as np

from .reference_fit import SCALES, jacobian, predict

LIMITS = np.array([0.05] * 3 + [np.deg2rad(5)] * 3 + [0.05])


def stationarity(j, residual):
    step, _, rank, singular = np.linalg.lstsq(j, residual, rcond=1e-7)
    sse = float(residual @ residual)
    gradient = j.T @ residual
    denominator = np.linalg.norm(j, axis=0) * np.linalg.norm(residual)
    cosine = np.divide(abs(gradient), denominator, out=np.zeros(7), where=denominator > 0)
    linear_change = j @ step
    reduction = float(2 * residual @ linear_change - linear_change @ linear_change)
    return {
        "rank": int(rank),
        "singular_values": singular.tolist(),
        "condition_number": float(singular[0] / singular[-1]) if rank == 7 else None,
        "scaled_step_norm": float(np.linalg.norm(step)),
        "scaled_gradient_infinity_norm_m2": float(np.max(abs(gradient))),
        "max_residual_column_cosine": float(np.max(cosine)),
        "linear_predicted_reduction_m2": reduction,
        "linear_reduction_over_eps_sse": float(reduction / (np.finfo(float).eps * sse))
        if sse
        else None,
        "sse_m2": sse,
    }


def finite_difference_jacobian(rays, planes, parameters, h):
    if not np.isfinite(h) or h <= 0:
        raise ValueError("positive finite difference step required")
    columns = []
    for i in range(7):
        delta = np.zeros(7)
        delta[i] = h
        columns.append(
            (
                (
                    predict(rays, planes, parameters + delta)
                    - predict(rays, planes, parameters - delta)
                )
                / (2 * h)
            ).ravel()
            * SCALES[i]
        )
    return np.array(columns).T


def trace_reference_fit(rays, planes, observed):
    observed = np.asarray(observed, dtype=float)
    p = np.zeros(7)
    if observed.shape != predict(rays, planes, p).shape or not np.isfinite(observed).all():
        raise ValueError("finite matching observations required")
    status = "iteration_limit"
    trace = []
    for iteration in range(20):
        predicted = predict(rays, planes, p)
        residual = (observed - predicted).ravel()
        j = jacobian(rays, planes, p)
        step, _, rank, _ = np.linalg.lstsq(j, residual, rcond=1e-7)
        row = {
            "iteration": iteration + 1,
            "parameters": p.tolist(),
            "stationarity": stationarity(j, residual),
            "trials": [],
        }
        trace.append(row)
        if rank < 7:
            status = "unidentifiable"
            break
        if np.linalg.norm(step) < 1e-7:
            status = "converged"
            break
        accepted = False
        for backtrack in range(12):
            trial = p + SCALES * step / (2**backtrack)
            trial_record = {"backtrack": backtrack, "step_fraction": 2.0 ** (-backtrack)}
            row["trials"].append(trial_record)
            if np.any(np.abs(trial) > LIMITS):
                trial_record["outcome"] = "parameter_bounds"
                continue
            try:
                trial_predicted = predict(rays, planes, trial)
                trial_residual = (observed - trial_predicted).ravel()
            except ValueError:
                trial_record["outcome"] = "invalid_geometry"
                continue
            trial_sse = np.dot(trial_residual, trial_residual)
            sse = np.dot(residual, residual)
            # Diagnostic only: subtraction-free SSE change, for the same rounded predictions.
            prediction_change = (trial_predicted - predicted).ravel()
            stable_delta = float(
                -2 * residual @ prediction_change + prediction_change @ prediction_change
            )
            trial_record.update(
                {
                    "sse_change_m2": float(trial_sse - sse),
                    "stable_sse_change_m2": stable_delta,
                    "sse_change_over_eps_sse": float(
                        (trial_sse - sse) / (np.finfo(float).eps * sse)
                    )
                    if sse
                    else None,
                    "physical_step_norm": float(np.linalg.norm(trial - p)),
                }
            )
            if trial_sse <= sse:
                p = trial
                accepted = True
                trial_record["outcome"] = "accepted"
                break
            trial_record["outcome"] = "objective_increase"
        if not accepted:
            status = "stalled"
            break
    j = jacobian(rays, planes, p)
    residual = (observed - predict(rays, planes, p)).ravel()
    singular = np.linalg.svd(j, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-7))
    fit = {
        "status": status,
        "iterations": iteration + 1,
        "parameters": p.tolist(),
        "residual_rmse_m": float(np.sqrt(np.mean((observed - predict(rays, planes, p)) ** 2))),
        "rank": rank,
        "minimum_singular_value_per_sqrt_sample": float(singular[-1] / np.sqrt(observed.size)),
        "scaled_condition_number": float(singular[0] / singular[-1]) if rank == 7 else None,
        "calibration_certified": False,
    }
    return {
        "fit": fit,
        "trace": trace,
        "terminal": stationarity(j, residual),
        "max_parameter_bound_fraction": float(np.max(abs(p) / LIMITS)),
        "terminal_fd_sensitivity": [
            {"step": h, **stationarity(finite_difference_jacobian(rays, planes, p, h), residual)}
            for h in (1e-7, 1e-6, 1e-5)
        ],
    }
