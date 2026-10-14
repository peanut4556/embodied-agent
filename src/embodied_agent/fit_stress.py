"""Prespecified synthetic departures from the stationary homoscedastic reference model."""

import numpy as np


def stress_geometry(rays, planes, scenario, config):
    r = np.array(rays, copy=True)
    actual = [(np.array(n, copy=True), d) for n, d in planes]
    if scenario == "near_degenerate":
        r[:, :2] *= config["near_degenerate_ray_multiplier"]
    if scenario == "single_reference_shift":
        actual[0] = (actual[0][0], actual[0][1] + config["single_reference_shift_m"])
    if scenario == "coherent_reference_shift":
        translation = np.array(config["coherent_reference_translation_m"])
        actual = [(n, float(d + n @ translation)) for n, d in actual]
    return r, actual


def stress_observation(clean, rays, seed, scenario, phase, config):
    if scenario not in config["scenarios"] or phase not in ("calibration", "validation"):
        raise ValueError("known scenario and calibration/validation phase required")
    rng = np.random.default_rng(seed)
    # Draw all components in every scenario to preserve paired comparisons.
    pixel = rng.normal(size=clean.shape)
    exposure = rng.normal(size=(4, 1))
    outlier_mask = rng.random(clean.shape) < config["outlier_probability"]
    outlier_sign = rng.choice([-1.0, 1.0], size=clean.shape)
    burst_mask = rng.random((4, 1)) < config["validation_burst_probability"]
    pixel_scale, exposure_scale = 1.0, 1.0
    if scenario == "heteroscedastic":
        pixel_scale = 0.5 + 1.5 * (rays[:, 0] / np.max(abs(rays[:, 0]))) ** 2
        exposure_scale = np.array(config["hetero_exposure_multipliers"])[:, None]
    observed = (
        clean
        + pixel * config["pixel_sigma_m"] * pixel_scale
        + exposure * config["exposure_sigma_m"] * exposure_scale
    )
    if scenario == "pixel_outliers":
        observed += outlier_mask * outlier_sign * config["outlier_magnitude_m"]
    if scenario == "validation_bursts" and phase == "validation":
        observed += burst_mask * config["validation_burst_magnitude_m"]
    return observed
