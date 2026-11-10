import unittest

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.pipelined_challenge import PipelinedChallengeController

P = (0.64, 0, 0.045)


def create(**kwargs):
    return PipelinedChallengeController(
        0,
        ExposureClockMap(0, 0),
        "camera",
        position_error_m=(0.001,) * 3,
        delivery_budget_s=0.1,
        exposure_lateness_s=0.01,
        **kwargs,
    )


class PipelineJitterTests(unittest.TestCase):
    def test_late_poll_and_exposure_share_one_budget(self):
        c = create()
        req = c.poll(0.004, 0.004)
        p = StampedPosition(P, 0.01)
        c.complete(req.request_id, 0.08, 0.08, p, p, session_id="camera")
        self.assertEqual(c.status, "pending")
        self.assertEqual(c.check.rows[-1]["exposure_time"], 0.01)
        c = create()
        req = c.poll(0.004, 0.004)
        p = StampedPosition(P, 0.011)
        c.complete(req.request_id, 0.08, 0.08, p, p, session_id="camera")
        self.assertEqual(c.reason, "capture_not_at_scheduled_request")

    def test_no_catchup_or_pre_request_frame(self):
        c = create()
        self.assertIsNone(c.poll(0.011, 0.011))
        self.assertEqual(c.reason, "capture_schedule_missed")
        c = create()
        c.poll(0.005, 0.005)
        p = StampedPosition(P, 0.004)
        c.complete(1, 0.08, 0.08, p, p, session_id="camera")
        self.assertEqual(c.reason, "capture_not_at_scheduled_request")

    def test_window_uses_task_clock_units(self):
        c = create()
        c.check.clock_map = ExposureClockMap(10, 0, rate=2)
        c.poll(0, 10)
        p = StampedPosition(P, 10.006)
        c.complete(1, 0.08, 10.04, p, p, session_id="camera")
        self.assertEqual(c.reason, "capture_not_at_scheduled_request")

    def test_admitted_jitter_does_not_relax_evidence_cadence(self):
        c = create()
        c.poll(0, 0)
        p = StampedPosition(P, 0.005)
        c.complete(1, 0.08, 0.08, p, p, session_id="camera")
        c.poll(0.2, 0.2)
        p = StampedPosition(P, 0.2)
        c.complete(2, 0.28, 0.28, p, p, session_id="camera")
        self.assertEqual(c.reason, "exposure_rejected")
        self.assertIn("scheduled capture", c.check.rows[-1]["error"])

    def test_final_exposure_cannot_use_delivery_reserve(self):
        c = create()
        for index, due in enumerate(c.schedule):
            c.poll(due, due)
            p = StampedPosition(P, due if index < 30 else due + 0.001)
            c.complete(index + 1, due + 0.01, due + 0.01, p, p, session_id="camera")
        self.assertEqual(c.reason, "capture_not_at_scheduled_request")

    def test_validation(self):
        for value in (True, -0.001, 0.011, float("nan"), "0.01"):
            with self.assertRaises(ValueError):
                PipelinedChallengeController(
                    0,
                    ExposureClockMap(0, 0),
                    "camera",
                    position_error_m=(0.001,) * 3,
                    exposure_lateness_s=value,
                )
