import json
import unittest
from pathlib import Path

from scripts.evaluate_budget_drift import evaluate


class BudgetDriftTests(unittest.TestCase):
    def test_frozen_report_reproduces(self):
        self.assertEqual(
            evaluate(), json.loads(Path("docs/evaluations/budget-drift-v1.json").read_text())
        )

    def test_sticky_state_and_pairing(self):
        rows = evaluate()["rows"]
        self.assertEqual(len(rows), 1080)
        states = {}
        for index in range(0, len(rows), 3):
            fixed, reset, sticky = rows[index : index + 3]
            for key in ("duration_s", "remaining_s", "feasible"):
                self.assertEqual(fixed[key], reset[key])
                self.assertEqual(reset[key], sticky[key])
            self.assertFalse(reset["degraded_before"])
            key = sticky["scenario"], sticky["action"]
            self.assertEqual(sticky["degraded_before"], states.get(key, False))
            if sticky["degraded_before"]:
                self.assertTrue(sticky["degraded_after"])
                self.assertGreaterEqual(sticky["required_s"], 0.05)
            states[key] = sticky["degraded_after"]

    def test_remaining_risk_and_rejection_cost_are_visible(self):
        summary = evaluate()["summary"]
        self.assertLess(summary["sticky"]["unconfirmed"], summary["reset"]["unconfirmed"])
        self.assertGreater(
            summary["sticky"]["rejected_feasible"], summary["reset"]["rejected_feasible"]
        )
        self.assertGreater(summary["sticky"]["unconfirmed_after_degradation"], 0)
