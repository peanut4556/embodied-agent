"""Audit regressions preserve counterexamples, not certify the controller as safe."""

import unittest

from scripts.evaluate_sampling_aliases import displacement, run


class SamplingAliasTests(unittest.TestCase):
    def test_exact_alias_counterexamples_remain_visible(self):
        for profile, periods in (("strict", (25, 50, 100)), ("spaced", (27.5, 55, 110))):
            for period in periods:
                with self.subTest(profile=profile, period=period):
                    r = run(profile, period, 0, None)
                    self.assertEqual(r["outcome"], "false_stationarity_confirmation")
                    self.assertGreater(r["dense_truth_span_m"], 0.0039)
                    self.assertLess(r["observed_challenge_span_m"], 1e-10)
                    self.assertEqual(r["result"]["request_count"], 31)

    def test_observable_motion_is_distinct_from_timing_rejection(self):
        for profile in ("strict", "spaced"):
            r = run(profile, 200, 0, None)
            self.assertEqual(r["outcome"], "motion_rejection")
        r = run("strict", 200, 0, 202611115)
        self.assertEqual(r["outcome"], "timing_or_other_rejection")

    def test_static_controls_do_not_count_as_missed_motion(self):
        for profile in ("strict", "spaced"):
            r = run(profile, None, 0, None)
            self.assertEqual(r["outcome"], "static_confirmation")
            self.assertEqual(r["dense_truth_span_m"], 0)
        self.assertEqual(run("spaced", None, 0, 202611115)["outcome"], "static_confirmation")

    def test_seed_reproducibility_and_causal_exposures(self):
        a = run("spaced", 55, 0, 202611115)
        self.assertEqual(a, run("spaced", 55, 0, 202611115))
        self.assertEqual(a["outcome"], "motion_rejection")
        for frame in a["frames"]:
            self.assertLessEqual(frame["request_ms"], frame["exposure_ms"])
            self.assertLessEqual(frame["exposure_ms"] - frame["request_ms"], 5)
            self.assertEqual(frame["delivery_ms"] - frame["exposure_ms"], 80)

    def test_motion_is_absent_during_candidate_search(self):
        self.assertEqual(displacement(2.05, 55, 0.25), 0)
        self.assertAlmostEqual(displacement(2.105, 55, 0), 0)
        self.assertAlmostEqual(displacement(2.11875, 55, 0), 0.002)

    def test_unknown_profile_is_not_silently_spaced(self):
        with self.assertRaises(ValueError):
            run("unknown", 55, 0, None)
