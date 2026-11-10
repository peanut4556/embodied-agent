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
        **kwargs,
    )


def simulate(delay_ms, capacity=2, motion=False):
    c = create(capacity=capacity)
    jobs = {}
    for ms in range(3101):
        now = ms / 1000
        if ms in jobs:
            req = jobs.pop(ms)
            position = (0.642, 0, 0.045) if motion and req.task_time >= 2.05 else P
            p = StampedPosition(position, req.host_time)
            c.complete(req.request_id, now, now, p, p, session_id="camera")
        req = c.poll(now, now)
        if req:
            jobs[ms + delay_ms] = req
        if c.status != "pending":
            break
    return c


class PipelineTests(unittest.TestCase):
    def test_latency_capacity_matrix(self):
        for delay in (20, 50, 80, 100, 120):
            for capacity in (1, 2, 3):
                with self.subTest(delay=delay, capacity=capacity):
                    c = simulate(delay, capacity)
                    expected = delay <= 100 and capacity * 50 >= delay
                    self.assertEqual(c.status == "confirmed_in_tray", expected)
                    self.assertLessEqual(c.peak_inflight, capacity)
                    self.assertLessEqual(len(c.requests), 31)
                    self.assertTrue(all(r["task_time"] <= 3 for r in c.requests))
                    if expected:
                        self.assertEqual(len(c.requests), 31)
                        self.assertAlmostEqual(c.events[-1]["delivery_time"], 3 + delay / 1000)

    def test_motion_still_rejected(self):
        c = simulate(80, motion=True)
        self.assertEqual(c.reason, "challenge_motion")
        self.assertTrue(c.result()["driver_cancel_required"])

    def test_late_poll_never_catches_up(self):
        c = create()
        c.poll(0.001, 0.001)
        self.assertEqual(c.reason, "capture_schedule_missed")
        self.assertEqual(c.requests, [])

    def test_timeout_without_results(self):
        c = create()
        c.poll(0, 0)
        c.poll(0.2, 0.2)
        c.poll(0.25, 0.25)
        self.assertEqual(c.reason, "capture_request_timeout")
        self.assertEqual(c.result()["cancel_request_ids"], [1, 2])

    def test_out_of_order_and_terminal_quarantine(self):
        c = create()
        c.poll(0, 0)
        c.poll(0.2, 0.2)
        p = StampedPosition(P, 0.2)
        self.assertFalse(c.complete(2, 0.21, 0.21, p, p, session_id="camera"))
        self.assertEqual(c.reason, "out_of_order_completion")
        self.assertFalse(c.complete(1, 0.22, 0.22, p, p, session_id="camera"))
        self.assertEqual(len(c.pending), 2)

    def test_unknown_duplicate_stop_snapshot(self):
        c = create()
        req = c.poll(0, 0)
        p = StampedPosition(P, 0)
        self.assertFalse(c.complete(99, 0.01, 0.01, p, p, session_id="camera"))
        self.assertTrue(c.complete(req.request_id, 0.02, 0.02, p, p, session_id="camera"))
        self.assertFalse(c.complete(req.request_id, 0.03, 0.03, p, p, session_id="camera"))
        c.poll(0.2, 0.2)
        c.poll(0.21, 0.21, stop=True)
        r = c.result()
        self.assertEqual(r["status"], "cancelled")
        r["pending"].clear()
        self.assertEqual(len(c.pending), 1)
        self.assertIsNone(c.poll(0.4, 0.4))

    def test_frame_epoch_session_and_pair_validation(self):
        for stamp, session, secondary in (
            (0.01, "camera", True),
            (0, "other", True),
            (0, "camera", False),
        ):
            c = create()
            c.poll(0, 0)
            p = StampedPosition(P, stamp)
            c.complete(1, 0.02, 0.02, p, p if secondary else None, session_id=session)
            self.assertEqual(c.status, "unconfirmed")

    def test_clock_backwards_and_drift(self):
        c = create()
        c.poll(0, 0)
        before = c.result()
        with self.assertRaises(ValueError):
            c.poll(-1, -1)
        self.assertEqual(c.result(), before)
        c.poll(0.1, 0.15)
        self.assertEqual(c.reason, "controller_clock_mismatch")

    def test_capacity_validation(self):
        for capacity in (True, 0, -1, 32, 1.5):
            with self.assertRaises(ValueError):
                create(capacity=capacity)

    def test_frame_age_limit_is_not_relaxed(self):
        c = create()
        c.check.clock_map = ExposureClockMap(0, 0, max_age=0.05)
        c.poll(0, 0)
        p = StampedPosition(P, 0)
        c.complete(1, 0.08, 0.08, p, p, session_id="camera")
        self.assertEqual(c.reason, "exposure_rejected")
        self.assertIn("too old", c.check.result()["deliveries"][-1]["error"])
