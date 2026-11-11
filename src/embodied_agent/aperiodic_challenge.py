"""Deterministic unequal intervals, trading lateness tolerance for phase diversity."""

from .spaced_challenge import SpacedChallengeController

# Fixed before evaluation; each ten-interval block sums to 550 ms.
CHALLENGE_INTERVALS_MS = (52, 58, 53, 57, 54, 56, 55, 55, 52, 58) * 2


class AperiodicChallengeController(SpacedChallengeController):
    """31 requests and the same 3.16 s deadline, with only 0--2 ms lateness.

    Actual challenge gaps lie in 50--60 ms. This finite unequal schedule is not
    random and does not guarantee detection of every periodic signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exposure_lateness_s = 0.002
        offsets = list(range(0, 2051, 205))
        for interval in CHALLENGE_INTERVALS_MS:
            offsets.append(offsets[-1] + interval)
        self.schedule = tuple(self.check.start + ms / 1000 for ms in offsets)

    def result(self):
        result = super().result()
        result["timing_profile"] = "unequal_2ms"
        result["challenge_intervals_ms"] = list(CHALLENGE_INTERVALS_MS)
        return result
