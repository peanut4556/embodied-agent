"""Prespecified synthetic external-sensor departures; never used as an admission rule."""

import numpy as np


def external_observation(truth_translation, seed, position_in_batch, scenario):
    """Local trials: position is current; latency uses an explicitly assumed local velocity."""
    truth = np.asarray(truth_translation, dtype=float)
    if truth.shape != (3,) or not np.isfinite(truth).all() or not 0 <= position_in_batch <= 1:
        raise ValueError("finite translation and normalized batch time required")
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=3)
    spike = rng.random() < scenario.get("outlier_probability", 0.0)
    sign = rng.choice([-1.0, 1.0])
    bias = np.zeros(3)
    kind = scenario["kind"]
    if kind == "mount":
        bias = np.array(scenario["offset_m"])
    elif kind == "drift":
        bias = position_in_batch * np.array(scenario["end_offset_m"])
    elif kind == "frame":
        angle = scenario["rotation_y_rad"]
        rotation = np.array(
            [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        )
        # Misregistered world frame, then subtract the unchanged nominal camera origin.
        world = np.array(scenario["nominal_camera_origin_m"]) + truth
        bias = rotation @ world - world
    elif kind == "latency":
        bias = -scenario["delay_s"] * np.array(scenario["velocity_m_s"])
    elif kind == "outliers":
        bias[0] = spike * sign * scenario["outlier_magnitude_m"]
    elif kind != "gaussian":
        raise ValueError("unknown external scenario")
    measured = truth + bias + scenario["actual_sigma_m"] * noise
    return measured, bias
