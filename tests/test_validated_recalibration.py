import json
import unittest
from pathlib import Path

from embodied_agent.action_budget import ActionBudget
from scripts.evaluate_budget_recovery import evaluate


class ValidatedRecoveryTests(unittest.TestCase):
    def budget(self):
        budget = ActionBudget({"verify": [0.01] * 20}, drift_policy="quarantine")
        budget.record("verify", 0.04, confirmed=True)
        return budget

    def restore(self, budget, validation, validation_id="holdout"):
        return budget.validated_recalibrated(
            "verify",
            [0.01] * 20,
            batch_id="train",
            validation_samples=validation,
            validation_batch_id=validation_id,
        )

    def test_success_audits_without_fitting_validation(self):
        original = self.budget()
        updated = self.restore(original, [0.014] * 20)
        self.assertAlmostEqual(updated.required("verify"), 0.015)
        self.assertTrue(original.blocked("verify"))
        self.assertFalse(updated.blocked("verify"))
        audit = updated.recalibration_history()[0]
        self.assertEqual(audit["validation_sample_count"], 20)
        self.assertEqual(audit["validation_max_s"], 0.014)
        self.assertEqual(len(audit["validation_sha256"]), 64)

    def test_failed_validation_is_atomic(self):
        budget = self.budget()
        for values in ([0.016] * 20, [0.01] * 19, [float("nan")] * 20):
            with self.assertRaises(ValueError):
                self.restore(budget, values)
            self.assertEqual(budget.revision, 0)
            self.assertEqual(budget.recalibration_history(), [])
            self.assertTrue(budget.blocked("verify"))

    def test_distinct_batch_and_cross_role_reuse(self):
        for batch in ("train", "", " holdout ", "x" * 129):
            with self.assertRaises(ValueError):
                self.restore(self.budget(), [0.01] * 20, batch)
        budget = self.restore(self.budget(), [0.01] * 20)
        budget.record("verify", 0.04, confirmed=True)
        with self.assertRaises(ValueError):
            budget.recalibrated("verify", [0.01] * 20, batch_id="holdout")
        with self.assertRaises(ValueError):
            budget.validated_recalibrated(
                "verify",
                [0.01] * 20,
                batch_id="new",
                validation_samples=[0.01] * 20,
                validation_batch_id="train",
            )

    def test_invalid_runtime_measurement_rejected(self):
        budget = self.budget()
        for value in (True, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                budget.record("verify", value, confirmed=True)
        with self.assertRaises(ValueError):
            budget.record("verify", 0.01, confirmed="yes")

    def test_frozen_sequential_recovery(self):
        report = evaluate()
        self.assertEqual(
            report, json.loads(Path("docs/evaluations/budget-recovery-v1.json").read_text())
        )
        self.assertEqual(len(report["rows"]), 360)
        attempt = next(
            a
            for a in report["attempts"]
            if a["scenario"] == "still_slow" and a["mode"] == "validated"
        )
        self.assertFalse(attempt["accepted"])
        later = [
            r
            for r in report["rows"]
            if r["scenario"] == "still_slow" and r["mode"] == "validated" and r["index"] >= 10
        ]
        self.assertTrue(all(r["dispatched"] == 0 for r in later))
        drift = [
            r
            for r in report["rows"]
            if r["scenario"] == "second_drift" and r["mode"] == "validated" and r["index"] >= 26
        ]
        self.assertTrue(all(r["blocked_after"] and r["dispatched"] == 0 for r in drift))
