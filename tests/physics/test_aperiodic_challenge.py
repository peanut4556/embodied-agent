import unittest

from embodied_agent.aperiodic_challenge import (
    CHALLENGE_INTERVALS_MS,
    AperiodicChallengeController,
)
from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from scripts.evaluate_sampling_aliases import run


def create():
    return AperiodicChallengeController(
        0,
        ExposureClockMap(0, 0),
        "camera",
        position_error_m=(0.001,) * 3,
    )


class AperiodicChallengeTests(unittest.TestCase):
    def test_fixed_budget_and_gap_extremes(self):
        c = create()
        self.assertEqual(len(c.schedule), 31)
        self.assertEqual(sum(CHALLENGE_INTERVALS_MS), 1100)
        self.assertGreater(len(set(CHALLENGE_INTERVALS_MS)), 1)
        self.assertAlmostEqual(c.schedule[-1], 3.15)
        self.assertAlmostEqual(c.check.capture_deadline, 3.16)
        for gap in CHALLENGE_INTERVALS_MS:
            for a in (0, 2):
                for b in (0, 2):
                    self.assertTrue(50 <= gap + b - a <= 60)

    def test_saved_counterexamples_include_residual_misses(self):
        for period in (25, 27.5, 50, 55, 100, 110):
            for phase in (0, 0.5):
                with self.subTest(period=period, phase=phase):
                    r = run("unequal", period, phase, None, jitter_max_ms=2)
                    expected = (
                        "false_stationarity_confirmation"
                        if period in (55, 110)
                        else "motion_rejection"
                    )
                    self.assertEqual(r["outcome"], expected)

    def test_matched_static_controls(self):
        for seed in (None, 202611117, 202611118):
            r = run("unequal", None, 0, seed, jitter_max_ms=2)
            self.assertEqual(r["outcome"], "static_confirmation")
            self.assertEqual(r["result"]["request_count"], 31)
            self.assertLessEqual(r["result"]["peak_inflight"], 2)

    def test_larger_lateness_is_not_silently_accepted(self):
        for via_request in (True, False):
            c = create()
            c.poll(0.003 if via_request else 0, 0.003 if via_request else 0)
            if not via_request:
                p = StampedPosition((0.64, 0, 0.045), 0.003)
                c.complete(1, 0.08, 0.08, p, p, session_id="camera")
            self.assertEqual(c.status, "unconfirmed")

    def test_stop_quarantines_active_work(self):
        c = create()
        c.poll(0, 0)
        c.poll(0.01, 0.01, stop=True)
        self.assertEqual(c.result()["cancel_request_ids"], [1])
        p = StampedPosition((0.64, 0, 0.045), 0)
        self.assertFalse(c.complete(1, 0.08, 0.08, p, p, session_id="camera"))

    def test_auditable_profile_and_snapshot(self):
        c = create()
        r = c.result()
        self.assertEqual(r["timing_profile"], "unequal_2ms")
        self.assertEqual(r["exposure_lateness_s"], 0.002)
        r["challenge_intervals_ms"].clear()
        r["scheduled_exposures"].clear()
        self.assertEqual(len(c.result()["challenge_intervals_ms"]), 20)
        self.assertEqual(len(c.schedule), 31)
