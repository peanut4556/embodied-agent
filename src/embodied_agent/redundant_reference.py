"""A reference plane excluded from fitting tests model agreement, not absolute accuracy."""

import numpy as np

from .reference_fit import fit_reference, predict


def check_redundant_reference(rays, planes, observed):
    observed = np.asarray(observed, dtype=float)
    if len(planes) != 4 or observed.shape != (4, len(rays)) or not np.isfinite(observed).all():
        raise ValueError("four finite reference observations required")
    fit = fit_reference(rays, planes[:3], observed[:3])
    if fit["status"] != "converged":
        return {
            "status": "unknown",
            "fit": fit,
            "reason": "training_fit_not_converged",
            "calibration_certified": False,
        }
    residual = observed[3] - predict(rays, planes[3:], fit["parameters"])[0]
    p95 = float(np.quantile(np.abs(residual), 0.95))
    return {
        "status": "alarm" if p95 > 0.003 else "no_alarm",
        "fit": fit,
        "heldout_p95_m": p95,
        "heldout_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "heldout_signed_mean_m": float(np.mean(residual)),
        "calibration_certified": False,
    }
