"""Budgeted observation times that sweep phases of periodic disturbances."""

# Milliseconds, all gaps <=250 ms so the existing evidence continuity contract holds.
# Two 1.2 s sweeps plus 0.6 s tail: exactly 16 observations including endpoints.
PHASE_SWEEP_INTERVALS_MS = (152, 170, 190, 210, 230, 248) * 2 + (160, 200, 240)


def observation_times_ms(mode):
    """Return immutable relative capture times; no motion/truth input is consulted.

    This is an open-loop diagnostic schedule, not a real-time camera scheduler.
    It reduces phase locking for some periodic motions, without guaranteeing
    detection of arbitrary motion between frames.
    """
    if mode == "uniform_5hz":
        return tuple(range(0, 3001, 200))
    if mode == "uniform_20hz":
        return tuple(range(0, 3001, 50))
    if mode != "phase_sweep":
        raise ValueError("unknown observation schedule")
    times = [0]
    for interval in PHASE_SWEEP_INTERVALS_MS:
        times.append(times[-1] + interval)
    return tuple(times)
