import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import save_budget
from embodied_agent.operation_receipt import _write
from embodied_agent.operation_snapshot import export_snapshot
from embodied_agent.snapshot_diff import compare_snapshots


class SnapshotDiffTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "budget.json"
        self.budget = ActionBudget({"verify": [0.01] * 20}, drift_policy="quarantine")
        save_budget(self.budget, self.path)
        self.receipts = self.root / "budget.json.operations"
        self.receipts.mkdir()
        self.file = self.receipts / ("a" * 64 + ".json")
        self.pending = {"schema": 1, "request_sha256": "b" * 64, "state": "pending", "result": None}
        _write(self.file, self.pending)

    def snap(self, name):
        path = self.root / (name + ".json")
        export_snapshot(self.path, path)
        return path

    def complete(self):
        receipt = dict(self.pending)
        receipt.update(
            state="complete",
            result={
                "success": True,
                "message": "done",
                "completed_steps": 1,
                "confirmation_reports": [],
            },
        )
        _write(self.file, receipt)

    def test_identical_snapshot_is_unchanged(self):
        path = self.snap("before")
        report = compare_snapshots(path, path)
        self.assertEqual(report["counts"]["unchanged"], 1)
        self.assertFalse(report["needs_review"])
        self.assertFalse(report["same_source_verified"])

    def test_normal_completion_and_new_receipt(self):
        before = self.snap("before")
        self.complete()
        _write(self.receipts / ("c" * 64 + ".json"), self.pending)
        report = compare_snapshots(before, self.snap("after"))
        self.assertEqual(report["counts"]["changed"], 1)
        self.assertEqual(report["counts"]["added"], 1)
        self.assertFalse(report["needs_review"])

    def test_removed_receipt_requires_review(self):
        before = self.snap("before")
        self.file.unlink()
        report = compare_snapshots(before, self.snap("after"))
        self.assertIn("receipt_missing", report["changes"][0]["flags"])
        self.assertTrue(report["needs_review"])

    def test_request_rebinding_is_flagged(self):
        before = self.snap("before")
        changed = dict(self.pending, request_sha256="d" * 64)
        _write(self.file, changed)
        report = compare_snapshots(before, self.snap("after"))
        self.assertIn("request_rebound", report["changes"][0]["flags"])

    def test_terminal_regression_is_flagged(self):
        self.complete()
        before = self.snap("before")
        _write(self.file, self.pending)
        report = compare_snapshots(before, self.snap("after"))
        self.assertIn("terminal_receipt_changed", report["changes"][0]["flags"])

    def test_abandonment_requires_matching_predecessor(self):
        before = self.snap("before")
        digest = json.loads(self.file.read_text())["sha256"]
        closed = {
            "schema": 2,
            "request_sha256": "b" * 64,
            "state": "abandoned",
            "result": None,
            "resolution": {
                "reason": "cancelled",
                "actor": "operator",
                "evidence_ref": "review",
                "previous_receipt_sha256": digest,
            },
        }
        _write(self.file, closed)
        self.assertFalse(compare_snapshots(before, self.snap("valid"))["needs_review"])
        closed["resolution"]["previous_receipt_sha256"] = "0" * 64
        _write(self.file, closed)
        report = compare_snapshots(before, self.snap("invalid"))
        self.assertIn("abandonment_predecessor_mismatch", report["changes"][0]["flags"])

    def test_format_only_change_is_not_terminal_mutation(self):
        self.complete()
        before = self.snap("before")
        self.file.write_text(json.dumps(json.loads(self.file.read_text()), indent=4))
        report = compare_snapshots(before, self.snap("after"))
        self.assertEqual(report["counts"]["encoding_changed"], 1)
        self.assertFalse(report["needs_review"])

    def test_invalid_sources_remain_uncertain_even_if_same(self):
        self.file.write_text("{}")
        self.path.write_text("{}")
        before = self.snap("before")
        report = compare_snapshots(before, before)
        self.assertEqual(report["counts"]["uncertain"], 1)
        self.assertIn("budget_unreadable", report["budget_flags"])
        self.assertTrue(report["needs_review"])

    def test_revision_rollback_and_history_truncation(self):
        self.budget.record("verify", 0.04, confirmed=True)
        newer = self.budget.recalibrated("verify", [0.03] * 20, batch_id="batch-1")
        save_budget(newer, self.path)
        before = self.snap("before")
        save_budget(self.budget, self.path)
        report = compare_snapshots(before, self.snap("after"))
        self.assertIn("budget_revision_decreased", report["budget_flags"])
        self.assertIn("calibration_history_not_extended", report["budget_flags"])

    def test_valid_recalibration_is_normal_but_unexplained_clear_flagged(self):
        self.budget.record("verify", 0.04, confirmed=True)
        save_budget(self.budget, self.path)
        before = self.snap("before")
        restored = self.budget.recalibrated("verify", [0.03] * 20, batch_id="batch-1")
        save_budget(restored, self.path)
        self.assertFalse(compare_snapshots(before, self.snap("valid"))["needs_review"])
        self.budget.degraded.clear()
        save_budget(self.budget, self.path)
        report = compare_snapshots(before, self.snap("cleared"))
        self.assertIn("degradation_cleared_without_new_calibration", report["budget_flags"])

    def test_tampered_snapshot_rejected(self):
        before = self.snap("before")
        invalid = self.root / "invalid.json"
        invalid.write_text("{}")
        with self.assertRaises(ValueError):
            compare_snapshots(before, invalid)

    def test_cli_review_exit_and_no_input_mutation(self):
        before = self.snap("before")
        self.file.unlink()
        after = self.snap("after")
        originals = before.read_bytes(), after.read_bytes()
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "embodied_agent.operation_admin",
                "compare-snapshots",
                "--before",
                str(before),
                "--after",
                str(after),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(child.returncode, 1, child.stderr)
        self.assertTrue(json.loads(child.stdout)["result"]["needs_review"])
        self.assertEqual(originals, (before.read_bytes(), after.read_bytes()))

    def test_configuration_change_and_cli_validation_codes(self):
        before = self.snap("before")
        self.budget.margin_s = 0.01
        save_budget(self.budget, self.path)
        after = self.snap("after")
        self.assertIn(
            "budget_configuration_changed", compare_snapshots(before, after)["budget_flags"]
        )
        invalid = self.root / "bad.json"
        invalid.write_text("{}")
        for target, expected in ((before, 0), (invalid, 3)):
            child = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "embodied_agent.operation_admin",
                    "compare-snapshots",
                    "--before",
                    str(before),
                    "--after",
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(child.returncode, expected, child.stderr)
