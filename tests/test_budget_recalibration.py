import unittest

from embodied_agent.action_budget import ActionBudget
from embodied_agent.runtime import AgentRuntime
from scripts.evaluate_budget_drift import Fixture


class RecalibrationTests(unittest.TestCase):
    def budget(self):
        budget = ActionBudget(
            {"pick": [0.01] * 20, "verify": [0.01] * 20}, drift_policy="quarantine"
        )
        budget.record("pick", 0.04, confirmed=True)
        budget.record("verify", 0.03, confirmed=True)
        return budget

    def test_target_only_recovery_and_task_execution(self):
        original = self.budget()
        updated = original.recalibrated("verify", [0.03] * 20, batch_id="independent-001")
        self.assertTrue(original.blocked("verify"))
        self.assertTrue(updated.blocked("pick"))
        self.assertFalse(updated.blocked("verify"))
        fixture = Fixture("verify", 0.032, 0.08)
        result = AgentRuntime(fixture, fixture, fixture, clock=lambda: fixture.t).run(
            "test", task_timeout_s=0.1, action_budget=updated
        )
        self.assertTrue(result.success)
        self.assertEqual(result.confirmation_reports[0]["budget_revision"], 1)
        updated.record("verify", 0.06, confirmed=True)
        self.assertTrue(updated.blocked("verify"))

    def test_rejection_is_atomic_and_batch_ids_not_reused(self):
        original = self.budget()
        for samples in ([0.03] * 19, [float("nan")] * 20, [-1] * 20):
            with self.assertRaises(ValueError):
                original.recalibrated("verify", samples, batch_id="bad")
        self.assertEqual(original.revision, 0)
        self.assertTrue(original.blocked("verify"))
        updated = original.recalibrated("verify", [0.03] * 20, batch_id="batch-1")
        with self.assertRaises(ValueError):
            updated.recalibrated("pick", [0.03] * 20, batch_id="batch-1")
        with self.assertRaises(ValueError):
            updated.recalibrated("verify", [0.03] * 20, batch_id="batch-2")

    def test_audit_snapshot_is_detached(self):
        updated = self.budget().recalibrated("verify", [0.03] * 20, batch_id="batch-1")
        history = updated.recalibration_history()
        self.assertEqual(len(history[0]["samples_sha256"]), 64)
        self.assertEqual(history[0]["sample_count"], 20)
        history[0]["action"] = "pick"
        self.assertEqual(updated.recalibration_history()[0]["action"], "verify")
