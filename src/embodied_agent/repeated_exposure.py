"""Method-of-moments noise estimates from a stationary repeated-exposure batch."""

import numpy as np


def estimate_repeated_noise(observations):
    x = np.asarray(observations, dtype=float)
    if x.ndim != 3 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError(
            "finite repeat x exposure x pixel observations, each dimension >=2 required"
        )
    repeats, exposures, pixels = x.shape
    # Remove the unknown static depth at each pixel, then each exposure's mean.
    temporal = x - x.mean(axis=0, keepdims=True)
    residual = temporal - temporal.mean(axis=2, keepdims=True)
    pixel_variance = float(np.sum(residual**2) / (exposures * (repeats - 1) * (pixels - 1)))
    means = x.mean(axis=2)
    covariance = np.cov(means, rowvar=False, ddof=1)
    global_raw = float((covariance.sum() - np.trace(covariance)) / (exposures * (exposures - 1)))
    exposure_raw = float(np.trace(covariance) / exposures - global_raw - pixel_variance / pixels)
    return {
        "sigma_pixel_m": float(np.sqrt(pixel_variance)),
        "sigma_exposure_m": float(np.sqrt(max(0.0, exposure_raw))),
        "sigma_global_m": float(np.sqrt(max(0.0, global_raw))),
        "raw_variance_components_m2": {
            "pixel": pixel_variance,
            "exposure": exposure_raw,
            "global": global_raw,
        },
        "negative_components_clipped": [
            name
            for name, value in (("exposure", exposure_raw), ("global", global_raw))
            if value < 0
        ],
        "exposure_mean_covariance_m2": covariance.tolist(),
        "shape": list(x.shape),
        "calibration_certified": False,
        "assumptions": [
            "stationary geometry and pixel correspondence across repetitions",
            "equal pixel/exposure variances, global component shared within each repetition",
            "independent repetitions; static offsets are not identifiable from repeat differences",
        ],
    }
