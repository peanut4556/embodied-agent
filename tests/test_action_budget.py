import json
import unittest
from pathlib import Path

from embodied_agent.action_budget import ActionBudget
from scripts.evaluate_action_budget import evaluate


class ActionBudgetTests(unittest.TestCase):
    def test_sparse_and_unknown_fall_back(self):
        budget = ActionBudget({"pick": [0.02]})
        self.assertEqual(budget.required("pick"), 0.05)
        self.assertEqual(budget.required("verify"), 0.05)

    def test_drift_and_failure_stick_without_refitting(self):
        budget = ActionBudget({"verify": [0.01] * 20, "pick": [0.1] * 20})
        budget.record("verify", 0.03, confirmed=True)
        budget.record("verify", 0.001, confirmed=True)
        self.assertEqual(budget.required("verify"), 0.05)
        budget.record("pick", 0.01, confirmed=False)
        self.assertGreater(budget.required("pick"), 0.1)

    def test_invalid_samples(self):
        for value in (True, 0, -1, float("nan")):
            with self.assertRaises(ValueError):
                ActionBudget({"verify": [value]})

    def test_independent_validation_tradeoff(self):
        result = evaluate()
        self.assertEqual(
            result, json.loads(Path("docs/evaluations/action-budget-v1.json").read_text())
        )
        fixed, learned = result["summary"]["fixed"], result["summary"]["calibrated"]
        self.assertEqual(len(result["rows"]), 240)
        self.assertEqual(learned["unconfirmed"], 0)
        self.assertLess(learned["rejected_feasible"], fixed["rejected_feasible"])
        self.assertGreater(learned["successes"], fixed["successes"])
