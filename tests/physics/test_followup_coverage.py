import unittest

from test_process_post_confirmation import Session

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.followup_coverage import FollowupCoverage
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_post_confirmation import ProcessPostConfirmation
from scripts.audit_followup_stress import evaluate, run_case


class CoverageTests(unittest.TestCase):
    def test_intervals_are_not_double_counted_or_extended(self):
        coverage = FollowupCoverage(1, 3)
        for exposure in (1.125, 1.25, 1.375):
            coverage.record("valid", exposure)
        result = coverage.result("completed")
        self.assertEqual(result["covered_seconds"], 0.25)
        self.assertEqual(result["uncovered_seconds"], 1.75)
        self.assertEqual(result["coverage_fraction"], 0.125)
        self.assertEqual(result["longest_uncovered_seconds"], 1.625)
        result["counts"]["valid"] = 100
        self.assertEqual(coverage.result("completed")["counts"]["valid"], 3)

    def test_missing_rejected_and_expired_break_interval(self):
        for outcome in ("missing", "rejected", "expired"):
            coverage = FollowupCoverage(1, 3)
            coverage.record("valid", 1.125)
            coverage.record(outcome)
            coverage.record("valid", 1.25)
            self.assertEqual(coverage.result("completed")["covered_seconds"], 0)
            coverage.record("valid", 1.375)
            self.assertEqual(coverage.result("completed")["covered_seconds"], 0.125)

    def test_single_frame_and_long_gap_do_not_imply_coverage(self):
        coverage = FollowupCoverage(1, 3)
        self.assertEqual(coverage.result("completed")["assessment"], "no_valid_observations")
        coverage.record("valid", 1.25)
        coverage.record("valid", 1.75)
        self.assertEqual(coverage.result("completed")["covered_seconds"], 0)
        self.assertEqual(coverage.result("completed")["longest_uncovered_seconds"], 2)
        self.assertEqual(coverage.result("cancelled")["assessment"], "window_incomplete")

    def test_partial_window_uses_original_denominator(self):
        row = run_case(0.03125, "outside", "none", "stop")
        self.assertEqual(row["coverage"]["planned_seconds"], 2)
        self.assertFalse(row["coverage"]["window_completed"])
        self.assertEqual(row["coverage"]["assessment"], "window_incomplete")
        self.assertLess(row["coverage"]["coverage_fraction"], 0.25)
        self.assertIsNotNone(row["actual_alert_time"])

    def test_matrix_metrics_distinguish_all_missing_and_periodic_drop(self):
        report = evaluate()
        self.assertEqual(report["mismatches"], 0)
        for row in report["results"]:
            metrics = row["coverage"]
            self.assertGreaterEqual(metrics["coverage_fraction"], 0)
            self.assertLessEqual(metrics["coverage_fraction"], 1)
            if row["drops"] == "all":
                self.assertEqual(metrics["covered_seconds"], 0)
                self.assertEqual(metrics["assessment"], "no_valid_observations")
        full = run_case(0.125, "outside", "none", "none")["coverage"]
        dropped = run_case(0.125, "outside", "every_third", "none")["coverage"]
        self.assertLess(dropped["covered_seconds"], full["covered_seconds"])
        self.assertGreater(dropped["counts"]["missing"], 0)

    def test_polling_does_not_add_samples_and_expiry_counts_once(self):
        host = 17
        session = Session()
        adapter = ProcessPostConfirmation(
            session, {"status": "confirmed_in_tray"}, 1, ExposureClockMap(16, 0), clock=lambda: host
        )
        adapter.step(1)
        host = 17.125
        session.frame = StampedPosition((0.64, 0, 0.04), 17.109375)
        session.deliver = True
        adapter.step(1.125)
        for task in (1.15, 1.25, 1.4, 1.5):
            host = 16 + task
            adapter.step(task)
        metrics = adapter.result()["coverage"]
        self.assertEqual(metrics["counts"], {"valid": 1, "missing": 0, "rejected": 0, "expired": 1})
        self.assertEqual(metrics["covered_seconds"], 0)
        adapter.close()

    def test_invalid_or_repeated_exposure_does_not_corrupt_counts(self):
        coverage = FollowupCoverage(1, 3)
        coverage.record("valid", 1.25)
        for exposure in (1, 1.25, 3, float("nan")):
            with self.assertRaises(ValueError):
                coverage.record("valid", exposure)
        self.assertEqual(coverage.result("completed")["counts"]["valid"], 1)
