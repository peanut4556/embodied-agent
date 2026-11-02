import unittest

from scripts.audit_followup_stress import evaluate, reference_alert, run_case


class FollowupStressTests(unittest.TestCase):
    def test_complete_fault_matrix_matches(self):
        report = evaluate()
        self.assertEqual(report["cases"], 108)
        self.assertEqual(report["mismatches"], 0, [r for r in report["results"] if not r["match"]])
        self.assertEqual(
            report["status_counts"], {"completed": 18, "invalidated": 72, "cancelled": 18}
        )
        self.assertTrue(all(row["terminal_quiet"] and row["cleaned"] for row in report["results"]))

    def test_inside_transient_and_all_missing_do_not_alert(self):
        for row in evaluate()["results"]:
            if row["pattern"] != "outside" or row["drops"] == "all":
                self.assertIsNone(row["actual_alert_time"])

    def test_drop_frequency_can_remove_continuous_outside_evidence(self):
        uninterrupted = run_case(0.125, "outside", "none", "none")
        interrupted = run_case(0.125, "outside", "every_third", "none")
        self.assertIsNotNone(uninterrupted["actual_alert_time"])
        self.assertIsNone(interrupted["actual_alert_time"])
        self.assertGreater(interrupted["delivered_frames"], 3)

    def test_timeout_precedes_later_injected_fault(self):
        for fault in ("none", "clock_jump", "crash", "stop"):
            row = run_case(0.28125, "outside", "none", fault)
            self.assertEqual(row["actual_status"], "invalidated")
            self.assertEqual(row["actual_reason"], "camera_unavailable")
            self.assertEqual(row["delivered_frames"], 0)
            self.assertEqual(row["requests"], 1)

    def test_alert_history_survives_later_stop_or_fault(self):
        for fault in ("stop", "crash", "clock_jump"):
            row = run_case(0.03125, "outside", "none", fault)
            self.assertIsNotNone(row["actual_alert_time"])
            self.assertLess(row["actual_alert_time"], 1.5)
            self.assertTrue(row["terminal_quiet"])

    def test_reference_window_rejects_missing_inside_and_long_gap(self):
        outside = (0.7, 0, 0.04)
        rows = [{"task": t + 0.01, "exposure": t, "point": outside} for t in (1.1, 1.225, 1.35)]
        self.assertAlmostEqual(reference_alert(rows), 1.36)
        for point in (None, (0.64, 0, 0.04)):
            modified = [dict(row) for row in rows]
            modified[1]["point"] = point
            self.assertIsNone(reference_alert(modified))
        rows[-1]["exposure"] = 1.6
        self.assertIsNone(reference_alert(rows))
