import unittest

import test_recalibration_plan as fixtures

from embodied_agent.budget_checkpoint import save_budget
from embodied_agent.budget_recalibration_plan import diagnose_plan
from embodied_agent.checkpointed_runtime import checkpoint_lock


class DiagnosisTests(unittest.TestCase):
    setUp = fixtures.RecalibrationPlanTests.setUp
    write = fixtures.RecalibrationPlanTests.write
    cli = fixtures.RecalibrationPlanTests.cli

    def diagnose(self):
        before = self.path.read_bytes(), self.plan.read_bytes()
        result = diagnose_plan(self.path, self.plan)
        self.assertEqual(before, (self.path.read_bytes(), self.plan.read_bytes()))
        self.assertFalse(result["applied"])
        return result

    def test_ready_candidates_readonly(self):
        result = self.diagnose()
        self.assertTrue(result["ready_for_preview"])
        self.assertEqual(result["eligible_count"], 2)
        self.assertTrue(all(row["candidate_record"] for row in result["results"]))
        self.assertIsNone(result["combined_error"])

    def test_collect_multiple_actions_and_reasons(self):
        self.data_plan["requests"][0]["calibration"]["durations_s"] = [0.03] * 2
        self.data_plan["requests"][0]["validation"]["durations_s"] = [0.1] * 2
        self.budget.degraded.remove("pick")
        save_budget(self.budget, self.path)
        self.write()
        result = self.diagnose()
        self.assertFalse(result["ready_for_preview"])
        self.assertEqual(result["eligible_count"], 0)
        self.assertEqual(
            result["results"][0]["reasons"],
            [
                "insufficient_calibration_samples",
                "insufficient_validation_samples",
                "validation_exceeds_candidate",
            ],
        )
        self.assertEqual(result["results"][1]["reasons"], ["action_not_degraded"])

    def test_policy_failure_reported_for_every_action(self):
        self.budget.drift_policy = "fallback"
        save_budget(self.budget, self.path)
        result = self.diagnose()
        self.assertTrue(
            all("quarantine_policy_required" in row["reasons"] for row in result["results"])
        )

    def test_historical_batch_conflict(self):
        budget = self.budget.validated_recalibrated(
            "verify",
            [0.03] * 20,
            batch_id="cal-1",
            validation_samples=[0.032] * 20,
            validation_batch_id="val-1",
        )
        budget.degraded.add("verify")
        save_budget(budget, self.path)
        result = self.diagnose()
        self.assertEqual(result["results"][0]["reasons"], ["historical_batch_conflict"])
        self.assertTrue(result["results"][1]["eligible"])

    def test_combined_capacity_failure(self):
        self.budget.estimates = {f"other-{i}": 0.02 for i in range(127)}
        save_budget(self.budget, self.path)
        result = self.diagnose()
        self.assertEqual(result["eligible_count"], 2)
        self.assertFalse(result["ready_for_preview"])
        self.assertEqual(result["combined_error"], "combined_candidate_not_persistable")

    def test_individual_capacity_failure(self):
        self.budget.estimates = {f"other-{i}": 0.02 for i in range(128)}
        save_budget(self.budget, self.path)
        result = self.diagnose()
        self.assertEqual(result["eligible_count"], 0)
        self.assertTrue(
            all(row["reasons"] == ["candidate_not_persistable"] for row in result["results"])
        )

    def test_cli_ready_and_ineligible_exit_codes(self):
        self.assertEqual(self.cli("diagnose").returncode, 0)
        self.data_plan["requests"][1]["validation"]["durations_s"] = [0.1] * 20
        self.write()
        child = self.cli("diagnose")
        self.assertEqual(child.returncode, 1, child.stderr)
        self.assertEqual(child.stderr, "")
        self.assertIn("validation_exceeds_candidate", child.stdout)

    def test_structural_errors_and_lock_remain_hard_errors(self):
        with checkpoint_lock(self.path):
            self.assertEqual(self.cli("diagnose").returncode, 3)
        self.data_plan["requests"][1]["action"] = "verify"
        self.write()
        child = self.cli("diagnose")
        self.assertEqual(child.returncode, 3)
        self.assertEqual(child.stdout, "")
