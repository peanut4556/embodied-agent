import json
import unittest
from pathlib import Path

from scripts.evaluate_task_budget import evaluate


class TaskBudgetExperimentTests(unittest.TestCase):
    def test_frozen_report_reproduces(self):
        frozen = json.loads(Path("docs/evaluations/task-budget-v1.json").read_text())
        self.assertEqual(evaluate(), frozen)

    def test_paired_profiles_and_bounds(self):
        report = evaluate()
        rows = report["records"]
        self.assertEqual(len(rows), 240)
        for original, bounded in zip(rows[::2], rows[1::2], strict=True):
            self.assertEqual(
                (original["kind"], original["seed"]), (bounded["kind"], bounded["seed"])
            )
            self.assertLessEqual(len(bounded["dispatches_s"]), len(original["dispatches_s"]))
            self.assertEqual(bounded["late_dispatches"], 0)
            self.assertLessEqual(bounded["elapsed_s"], original["elapsed_s"])
        fast = [r for r in rows if r["kind"] == "fast"]
        self.assertTrue(all(r["success"] and r["completed"] == 4 for r in fast))
        slow = [r for r in rows if r["kind"] != "fast" and r["budget_s"] is not None]
        self.assertTrue(all(not r["success"] for r in slow))
        self.assertGreater(max(r["overshoot_s"] for r in slow), 0.1)
