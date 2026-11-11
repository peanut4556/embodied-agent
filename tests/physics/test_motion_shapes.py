import unittest

from scripts.evaluate_motion_shapes import evaluate, motion_value
from scripts.evaluate_sampling_aliases import run


class MotionShapeTests(unittest.TestCase):
    def test_waveform_amplitude_and_onset(self):
        for wave in ("sine", "triangle", "pulse"):
            self.assertEqual(motion_value(2.059, wave, 2.06, 100, 0), 0)
            values = [motion_value(2.06 + i / 1000, wave, 2.06, 100, 0) for i in range(101)]
            self.assertLessEqual(max(values), 0.002 + 1e-12)
            self.assertGreaterEqual(min(values), -0.002 - 1e-12)
        self.assertAlmostEqual(motion_value(2.085, "sine", 2.06, 100, 0), 0.002)

    def test_pulse_duty_cycle_and_phase(self):
        values = [motion_value(i / 1000, "pulse", 0, 100, 0) for i in range(100)]
        self.assertEqual(sum(v > 0 for v in values), 10)
        self.assertEqual(motion_value(0, "pulse", 0, 100, 0.5), 0)
        self.assertEqual(motion_value(0.05, "pulse", 0, 100, 0.5), 0.002)

    def test_post_exposure_motion_is_not_retroactive_evidence_failure(self):
        r = evaluate("sweep", "sine", 3.17, 55, 0, None)
        self.assertEqual(r["outcome"], "static_confirmation")
        self.assertEqual(r["dense_truth_span_m"], 0)
        self.assertTrue(r["confirmed_with_post_exposure_motion"])
        self.assertGreater(r["post_exposure_span_m"], 0.003)
        self.assertAlmostEqual(r["last_exposure_s"], 3.15)

    def test_early_motion_is_observed_before_delivery(self):
        r = evaluate("sweep", "sine", 2.06, 197, 0, None)
        self.assertEqual(r["outcome"], "motion_rejection")
        self.assertFalse(r["confirmed_with_post_exposure_motion"])

    def test_default_sine_audit_remains_unchanged(self):
        r = run("unequal", 55, 0, None, jitter_max_ms=2)
        self.assertEqual(r["outcome"], "false_stationarity_confirmation")
        r = run("sweep", 55, 0, None, jitter_max_ms=2)
        self.assertEqual(r["outcome"], "motion_rejection")

    def test_seeded_runs_are_reproducible(self):
        args = ("sweep", "triangle", 2.5, 83, 0.25, 202611121)
        self.assertEqual(evaluate(*args), evaluate(*args))
