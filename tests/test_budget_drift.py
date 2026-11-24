import json
import unittest
from pathlib import Path

from scripts.evaluate_budget_drift import evaluate


class BudgetDriftTests(unittest.TestCase):
    def test_frozen_report_reproduces(self):
        current = evaluate()
        frozen = json.loads(Path("docs/evaluations/budget-drift-v2.json").read_text())
        current.pop("sources_sha256")
        frozen.pop("sources_sha256")
        self.assertEqual(current, frozen)

    def test_sticky_state_and_pairing(self):
        rows = evaluate()["rows"]
        self.assertEqual(len(rows), 1440)
        states = {}
        for index in range(0, len(rows), 4):
            fixed, reset, sticky, quarantine = rows[index : index + 4]
            if quarantine["degraded_before"]:
                self.assertEqual(quarantine["dispatched"], 0)
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

    def test_original_profiles_unchanged(self):
        old = json.loads(Path("docs/evaluations/budget-drift-v1.json").read_text())
        new = evaluate()
        self.assertEqual(old["rows"], [r for r in new["rows"] if r["mode"] != "quarantine"])
        self.assertEqual(new["summary"]["quarantine"]["unconfirmed_after_degradation"], 0)
