"""Descriptive, time-aligned trajectory diagnostics; never controller inputs."""

import numpy as np


def first_sustained(values, threshold, count=3):
    """Return onset, not confirmation tick. None means no qualifying interval."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or count < 1:
        raise ValueError("invalid divergence series")
    run = 0
    for tick, value in enumerate(values):
        run = run + 1 if value > threshold else 0
        if run == count:
            return tick - count + 1
    return None


def events(trace, fps):
    def first(predicate):
        return next((r["tick"] for r in trace if predicate(r)), None)

    return {
        "first_closing_command_tick": first(lambda r: max(r["command"][3:]) < 0.01),
        "first_finger_contact_tick": first(lambda r: r["finger_contacts"] > 0),
        "first_bilateral_contact_tick": first(lambda r: r["finger_contacts"] == 2),
        "first_holding_tick": first(lambda r: r["holding"]),
        "contact_frames": sum(r["finger_contacts"] > 0 for r in trace),
        "bilateral_frames": sum(r["finger_contacts"] == 2 for r in trace),
        "fps": fps,
    }


def compare(parent, candidate):
    if not parent or not candidate:
        raise ValueError("empty trace")
    n = min(len(parent), len(candidate))
    if any(rows[t]["tick"] != t for rows in (parent, candidate) for t in range(len(rows))):
        raise ValueError("nonconsecutive trace ticks")
    result = {}
    for key in ("command", "joints"):
        delta = np.asarray([r[key] for r in candidate[:n]]) - np.asarray(
            [r[key] for r in parent[:n]]
        )
        result[f"{key}_arm_max_rad"] = np.max(np.abs(delta[:, :3]), axis=1)
        result[f"{key}_finger_max_m"] = np.max(np.abs(delta[:, 3:]), axis=1)
    result["object_distance_m"] = np.linalg.norm(
        np.asarray([r["block_xyz"] for r in candidate[:n]])
        - np.asarray([r["block_xyz"] for r in parent[:n]]),
        axis=1,
    )
    thresholds = {k: (1e-4 if "finger" in k else 1e-3) for k in result}
    return {
        "aligned_frames": n,
        "thresholds": thresholds,
        "first_numerical_tick": {k: first_sustained(v, 1e-8, 1) for k, v in result.items()},
        "first_sustained_tick": {k: first_sustained(v, thresholds[k]) for k, v in result.items()},
        "series": {k: v.tolist() for k, v in result.items()},
    }


def action_decomposition(parent_output, same_history_output, own_history_output):
    """Exact algebraic decomposition; the history term includes recurrent history."""
    p, same, own = (np.asarray(a) for a in (parent_output, same_history_output, own_history_output))
    if p.shape != same.shape or p.shape != own.shape or p.ndim != 2 or p.shape[1] != 5:
        raise ValueError("action histories must align")
    if not all(np.isfinite(a).all() for a in (p, same, own)):
        raise ValueError("nonfinite action histories")
    return {"weight_residual": same - p, "history_shift": own - same, "total": own - p}
