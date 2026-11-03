import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embodied_agent.coverage_audit import audit_coverage
from embodied_agent.followup_coverage import FollowupCoverage
from scripts.audit_followup_coverage import audit_matrix
from scripts.audit_followup_stress import evaluate


class CoverageAuditTests(unittest.TestCase):
    def run_cli(self, source, output):
        return subprocess.run(
            [
                sys.executable,
                "scripts/audit_followup_coverage.py",
                str(source),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_envelope_rejects_truncation_and_wrong_types(self):
        for key, value in (
            ("cases", 107),
            ("cases", True),
            ("schema_version", True),
            ("schema_version", 2),
            ("test_kind", "other"),
        ):
            report = evaluate()
            report[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                audit_matrix(report)
        report = evaluate()
        report["results"].pop()
        with self.assertRaises(ValueError):
            audit_matrix(report)

    def test_unknown_or_nonterminal_status_is_rejected_per_row(self):
        for status in (None, "unknown", "monitoring", [], True):
            report = evaluate()
            report["results"][0]["actual_status"] = status
            self.assertEqual(audit_matrix(report)["counts"], {"rejected": 1, "verified": 107})

    def test_missing_evidence_does_not_bypass_schema(self):
        for metrics in ({}, {"schema_version": 2}, {"schema_version": True}):
            with self.assertRaises(ValueError):
                audit_coverage(metrics)

    def test_boolean_exposure_and_overflowing_duration_rejected(self):
        coverage = FollowupCoverage(0, 2)
        coverage.record("valid", 1)
        for key in ("first_valid_exposure", "last_valid_exposure"):
            metrics = coverage.result("completed")
            metrics[key] = True
            with self.assertRaises(ValueError):
                audit_coverage(metrics)
        metrics = self.sample()
        metrics["evidence"].update(start=-1e308, deadline=1e308)
        with self.assertRaises(ValueError):
            audit_coverage(metrics)

    def test_cli_rejects_ambiguous_and_invalid_input_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.json", Path(directory) / "audit.json"
            for raw in (
                '{"results": [], "results": []}',
                '{"nested":{"x":1,"x":2}}',
                '{"x":NaN}',
                '{"x":Infinity}',
                '{"x":1e999}',
                "{",
                "{}",
            ):
                source.write_text(raw)
                result = self.run_cli(source, output)
                self.assertEqual(result.returncode, 3, result.stderr)
                self.assertIn("input rejected", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(output.exists())
            source.write_bytes(b"\xff")
            self.assertEqual(self.run_cli(source, output).returncode, 3)
            self.assertEqual(self.run_cli(Path(directory) / "missing", output).returncode, 3)

    def test_cli_exit_codes_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            report = evaluate()
            for code in (0, 1, 2):
                candidate = copy.deepcopy(report)
                if code == 1:
                    candidate["results"][0]["coverage"]["covered_seconds"] = -1
                if code == 2:
                    del candidate["results"][0]["coverage"]["evidence"]
                source.write_text(json.dumps(candidate))
                output = Path(directory) / f"audit-{code}.json"
                result = self.run_cli(source, output)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(len(json.loads(output.read_text())["results"]), 108)
                original = output.read_bytes()
                self.assertEqual(self.run_cli(source, output).returncode, 4)
                self.assertEqual(output.read_bytes(), original)

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
