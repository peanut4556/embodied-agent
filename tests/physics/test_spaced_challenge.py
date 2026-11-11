import unittest

from embodied_agent.confirmation_challenge import ConfirmationChallenge
from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.spaced_challenge import SpacedChallengeController
from scripts.evaluate_spaced_challenge import run

P = (0.64, 0, 0.045)


def create():
    return SpacedChallengeController(0, ExposureClockMap(0, 0), "s", position_error_m=(0.001,) * 3)


class SpacedChallengeTests(unittest.TestCase):
    def test_static_jitter_and_motion_comparison(self):
        for jitter in ("zero", "alternating", "random", "constant5"):
            with self.subTest(jitter=jitter):
                c = run("spaced", jitter, "static", 42)["result"]
                self.assertEqual(c["status"], "confirmed_in_tray")
                self.assertEqual(c["request_count"], 31)
                for scene in ("step", "periodic_after"):
                    c = run("spaced", jitter, scene, 42)["result"]
                    self.assertEqual(c["reason"], "challenge_motion")

    def test_schedule_geometry_all_jitter_extremes(self):
        c = create()
        self.assertAlmostEqual(c.check.capture_deadline, 3.16)
        self.assertAlmostEqual(c.check.deadline, 3.26)
        self.assertEqual(len(c.schedule), 31)
        for i, (a, b) in enumerate(zip(c.schedule, c.schedule[1:])):
            for x in (0, 0.005):
                for y in (0, 0.005):
                    gap = b + y - a - x
                    self.assertGreaterEqual(gap + 1e-9, 0.2 if i < 10 else 0.05)
                    self.assertLessEqual(gap, (0.21 if i < 10 else 0.06) + 1e-9)

    def test_request_and_exposure_share_window(self):
        c = create()
        c.poll(0.003, 0.003)
        p = StampedPosition(P, 0.006)
        c.complete(1, 0.08, 0.08, p, p, session_id="s")
        self.assertEqual(c.reason, "capture_not_at_scheduled_request")
        c = create()
        c.poll(0.006, 0.006)
        self.assertEqual(c.reason, "capture_schedule_missed")

    def test_timeout_stop_and_late_result(self):
        for stop in (False, True):
            c = create()
            c.poll(0, 0)
            c.poll(0.205, 0.205)
            c.poll(0.25, 0.25, stop=stop)
            self.assertEqual(c.reason, "user_stop" if stop else "capture_request_timeout")
            self.assertEqual(c.result()["cancel_request_ids"], [1, 2])
            p = StampedPosition(P, 0)
            self.assertFalse(c.complete(1, 0.26, 0.26, p, p, session_id="s"))

    def test_out_of_order_rejected(self):
        c = create()
        c.poll(0, 0)
        c.poll(0.205, 0.205)
        p = StampedPosition(P, 0.205)
        self.assertFalse(c.complete(2, 0.21, 0.21, p, p, session_id="s"))
        self.assertEqual(c.reason, "out_of_order_completion")

    def test_delivery_deadline_without_callback(self):
        c = create()
        c.poll(3.26, 3.26)
        self.assertEqual(c.reason, "delivery_deadline_without_complete_evidence")
        self.assertIsNone(c.poll(3.27, 3.27))

    def test_default_deadline_and_budget_validation(self):
        c = ConfirmationChallenge(0, position_error_m=(0.001,) * 3)
        self.assertEqual(c.deadline, 3)
        for value in (True, -0.1, 0.251, float("nan"), "0.16"):
            with self.assertRaises(ValueError):
                ConfirmationChallenge(0, position_error_m=(0.001,) * 3, capture_budget_s=value)

    def test_final_exposure_window_not_delivery_window(self):
        c = create()
        for index, due in enumerate(c.schedule):
            c.poll(due, due)
            p = StampedPosition(P, due + (0.011 if index == 30 else 0))
            c.complete(index + 1, due + 0.02, due + 0.02, p, p, session_id="s")
        self.assertEqual(c.reason, "capture_not_at_scheduled_request")
