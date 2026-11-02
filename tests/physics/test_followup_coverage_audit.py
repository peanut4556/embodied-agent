import copy
import json
import unittest
from pathlib import Path

from embodied_agent.coverage_audit import audit_coverage
from embodied_agent.followup_coverage import FollowupCoverage
from scripts.audit_followup_coverage import audit_matrix
from scripts.audit_followup_stress import evaluate


class CoverageAuditTests(unittest.TestCase):
    def sample(self):
        coverage = FollowupCoverage(1, 3)
        for outcome, exposure in (
            ("valid", 1.125),
            ("valid", 1.25),
            ("missing", None),
            ("valid", 1.5),
            ("valid", 1.625),
        ):
            coverage.record(outcome, exposure)
        return coverage.result("completed")

    def test_independent_union_and_json_round_trip(self):
        metrics = json.loads(json.dumps(self.sample()))
        replay = audit_coverage(metrics, status="completed")
        self.assertEqual(replay["intervals"], [[1.125, 1.25], [1.5, 1.625]])
        self.assertEqual(replay["covered_seconds"], 0.25)
        self.assertEqual(replay["longest_uncovered_seconds"], 1.375)

    def test_all_stress_scenarios_recompute(self):
        report = audit_matrix(evaluate())
        self.assertEqual(report["counts"], {"verified": 108})

    def test_changed_summary_is_rejected(self):
        for key, value in (
            ("coverage_fraction", 1),
            ("covered_seconds", 2),
            ("uncovered_seconds", 0),
            ("longest_uncovered_seconds", 0),
            ("planned_seconds", 0.25),
            ("assessment", "window_bracketed"),
        ):
            metrics = self.sample()
            metrics[key] = value
            with self.assertRaises(ValueError):
                audit_coverage(metrics)

    def test_deleted_missing_record_cannot_bridge_gap(self):
        metrics = self.sample()
        del metrics["evidence"]["records"][2]
        metrics["counts"]["missing"] = 0
        with self.assertRaisesRegex(ValueError, "covered_seconds"):
            audit_coverage(metrics)

    def test_ledger_is_detached_and_order_is_checked(self):
        coverage = FollowupCoverage(1, 3)
        coverage.record("valid", 1.25)
        metrics = coverage.result("completed")
        metrics["evidence"]["records"][0]["exposure"] = 2.5
        self.assertEqual(coverage.result("completed")["last_valid_exposure"], 1.25)
        metrics = self.sample()
        metrics["evidence"]["records"][1]["exposure"] = 1.125
        with self.assertRaises(ValueError):
            audit_coverage(metrics)

    def test_legacy_and_early_termination_remain_distinct(self):
        root = Path(__file__).resolve().parents[2]
        legacy = json.loads((root / "docs/evaluations/followup-coverage-v1.json").read_text())
        self.assertEqual(audit_matrix(legacy)["counts"], {"unavailable": 108})
        with self.assertRaises(ValueError):
            audit_coverage(self.sample(), status="cancelled")

    def test_one_bad_scenario_does_not_hide_valid_scenarios(self):
        report = evaluate()
        report["results"][0]["coverage"]["counts"]["valid"] = True
        audit = audit_matrix(report)
        self.assertEqual(audit["counts"], {"rejected": 1, "verified": 107})

    def test_invalid_schema_nonfinite_and_window_rejected(self):
        for start, end in ((1, 1), (3, 1), (float("nan"), 3)):
            with self.assertRaises(ValueError):
                FollowupCoverage(start, end)
        for value in (None, {}, {"schema_version": 2}):
            metrics = self.sample()
            metrics["evidence"] = copy.deepcopy(value)
            with self.assertRaises(ValueError):
                audit_coverage(metrics)
        metrics = self.sample()
        metrics["covered_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            audit_coverage(metrics)
