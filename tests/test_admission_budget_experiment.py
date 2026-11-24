import json
import unittest
from pathlib import Path

from scripts.evaluate_admission_budget import RESERVES, evaluate


class AdmissionExperimentTests(unittest.TestCase):
    def test_frozen_report_reproduces(self):
        self.assertEqual(
            evaluate(), json.loads(Path("docs/evaluations/admission-budget-v1.json").read_text())
        )

    def test_pairing_and_independent_feasibility(self):
        report = evaluate()
        self.assertEqual(len(report["records"]), 400)
        for spec in report["cases"]:
            total = spec["planning"] + sum(
                sum(spec[k]) for k in ("selection", "execution", "confirmation")
            )
            self.assertEqual(spec["feasible"], total < 0.2)
            rows = [
                r
                for r in report["records"]
                if (r["kind"], r["seed"]) == (spec["kind"], spec["seed"])
            ]
            self.assertEqual([r["reserve_s"] for r in rows], list(RESERVES))
            self.assertEqual(rows[0]["success"], spec["feasible"])
            self.assertTrue(all(r["late_dispatches"] == 0 for r in rows))

    def test_tradeoff_is_not_hidden(self):
        summary = evaluate()["summary"]
        self.assertEqual(summary["0"]["rejected_feasible"], 0)
        self.assertGreater(summary["0"]["unconfirmed_actions"], 0)
        self.assertEqual(summary["0.05"]["unconfirmed_actions"], 0)
        self.assertGreater(summary["0.05"]["rejected_feasible"], 0)
        self.assertEqual(summary["0.2"]["successes"], 0)
