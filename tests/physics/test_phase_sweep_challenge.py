import unittest

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.phase_sweep_challenge import (
    SWEEP_INTERVALS_MS,
    PhaseSweepChallengeController,
)
from scripts.evaluate_phase_sweep import HOLDOUT_PERIODS_MS, HOLDOUT_PHASES
from scripts.evaluate_sampling_aliases import run
from scripts.evaluate_unequal_sampling import PERIODS_MS


def create():
    return PhaseSweepChallengeController(
        0,
        ExposureClockMap(0, 0),
        "camera",
        position_error_m=(0.001,) * 3,
    )


class PhaseSweepTests(unittest.TestCase):
    def test_phase_excursion_and_unchanged_budget(self):
        c = create()
        self.assertEqual(sum(SWEEP_INTERVALS_MS), 1100)
        self.assertEqual(len(c.schedule), 31)
        offset = 0
        offsets = []
        for gap in SWEEP_INTERVALS_MS:
            offset += gap - 55
            offsets.append(offset)
            for a in (0, 2):
                for b in (0, 2):
                    self.assertTrue(50 <= gap + b - a <= 60)
        self.assertEqual(min(offsets), -30)
        self.assertEqual(offset, 0)
        self.assertAlmostEqual(c.schedule[-1], 3.15)
        self.assertAlmostEqual(c.check.capture_deadline, 3.16)

    def test_holdout_coordinates_disjoint_from_design_grid(self):
        self.assertFalse(set(HOLDOUT_PERIODS_MS) & set(PERIODS_MS))
        self.assertFalse(set(HOLDOUT_PHASES) & {i / 16 for i in range(16)})

    def test_remaining_zero_jitter_alias_witnesses(self):
        for period in (55, 110):
            for phase in (0, 0.5):
                with self.subTest(period=period, phase=phase):
                    self.assertEqual(
                        run("unequal", period, phase, None, jitter_max_ms=2)["outcome"],
                        "false_stationarity_confirmation",
                    )
                    self.assertEqual(
                        run("sweep", period, phase, None, jitter_max_ms=2)["outcome"],
                        "motion_rejection",
                    )

    def test_static_controls_with_new_seeds(self):
        for seed in (None, 202611119, 202611120):
            r = run("sweep", None, 0, seed, jitter_max_ms=2)
            self.assertEqual(r["outcome"], "static_confirmation")
            self.assertEqual(r["result"]["request_count"], 31)
            self.assertLessEqual(r["result"]["peak_inflight"], 2)

    def test_lateness_and_terminal_quarantine(self):
        c = create()
        c.poll(0, 0)
        p = StampedPosition((0.64, 0, 0.045), 0.003)
        c.complete(1, 0.08, 0.08, p, p, session_id="camera")
        self.assertEqual(c.reason, "capture_not_at_scheduled_request")
        self.assertIsNone(c.poll(0.205, 0.205))
        c = create()
        c.poll(0, 0)
        c.poll(0.01, 0.01, stop=True)
        self.assertEqual(c.result()["cancel_request_ids"], [1])
        self.assertFalse(c.complete(1, 0.08, 0.08, p, p, session_id="camera"))

    def test_snapshot_describes_actual_schedule(self):
        c = create()
        r = c.result()
        self.assertEqual(r["timing_profile"], "phase_sweep_2ms")
        self.assertEqual(r["challenge_intervals_ms"], list(SWEEP_INTERVALS_MS))
        r["challenge_intervals_ms"].clear()
        self.assertEqual(len(c.result()["challenge_intervals_ms"]), 20)
