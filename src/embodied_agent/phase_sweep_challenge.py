"""Accumulate then return timing phase within the existing acquisition budget."""

from .aperiodic_challenge import AperiodicChallengeController

SWEEP_INTERVALS_MS = (52,) * 10 + (58,) * 10


class PhaseSweepChallengeController(AperiodicChallengeController):
    """Fixed two-stage interval sweep, with the same 0--2 ms lateness contract.

    Relative to 55 ms spacing, cumulative offsets reach -30 ms before returning
    to zero. This is an experimental finite schedule, not an anti-alias guarantee.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        offsets = list(range(0, 2051, 205))
        for interval in SWEEP_INTERVALS_MS:
            offsets.append(offsets[-1] + interval)
        self.schedule = tuple(self.check.start + ms / 1000 for ms in offsets)

    def result(self):
        result = super().result()
        result["timing_profile"] = "phase_sweep_2ms"
        result["challenge_intervals_ms"] = list(SWEEP_INTERVALS_MS)
        return result
