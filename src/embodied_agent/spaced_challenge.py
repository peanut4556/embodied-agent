"""Explicitly longer acquisition protocol with five milliseconds of jitter room."""

from .delivered_challenge import DeliveredChallenge
from .pipelined_challenge import PipelinedChallengeController


class SpacedChallengeController(PipelinedChallengeController):
    """Fixed 31 requests: ten 205 ms intervals followed by twenty 55 ms intervals.

    Every exposure may be 0--5 ms late relative to its slot. Thus actual search
    gaps are 200--210 ms and challenge gaps 50--60 ms. Exposure deadline is
    explicitly 3.16 s, not the strict protocol's 3 s. No driver is started.
    """

    def __init__(
        self,
        start_time,
        clock_map,
        session_id,
        *,
        position_error_m,
        capacity=2,
        delivery_budget_s=0.1,
        request_timeout_s=0.25,
    ):
        super().__init__(
            start_time,
            clock_map,
            session_id,
            position_error_m=position_error_m,
            capacity=capacity,
            exposure_lateness_s=0.005,
            delivery_budget_s=delivery_budget_s,
            request_timeout_s=request_timeout_s,
        )
        self.check = DeliveredChallenge(
            start_time,
            clock_map,
            session_id,
            position_error_m=position_error_m,
            delivery_budget_s=delivery_budget_s,
            capture_budget_s=0.16,
        )
        self.schedule = tuple(
            self.check.start + ms / 1000 for ms in (*range(0, 2051, 205), *range(2105, 3151, 55))
        )

    def result(self):
        result = super().result()
        result["timing_profile"] = "spaced_5ms"
        result["scheduled_exposures"] = list(self.schedule)
        return result
