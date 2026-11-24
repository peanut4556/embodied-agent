import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import save_budget
from embodied_agent.checkpointed_runtime import checkpoint_lock
from embodied_agent.operation_admin import abandon_operation, inspect_operation
from embodied_agent.operation_receipt import _write, read_receipt, receipt_path, run_operation


class OperationAdminTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "budget.json"
        save_budget(ActionBudget({}, drift_policy="quarantine"), self.path)
        self.runtime = Mock()
        self.runtime.run.side_effect = RuntimeError("failed")
        self.invoke()

    def invoke(self):
        return run_operation(
            self.runtime,
            "test",
            self.path,
            operation_id="op-1",
            task_timeout_s=1,
            request_context={},
        )

    def close(self, checksum=None):
        checksum = checksum or inspect_operation(self.path, "op-1")["receipt_sha256"]
        return abandon_operation(
            self.path,
            "op-1",
            expected_receipt_sha256=checksum,
            reason="operator cancelled",
            evidence_ref="local-review-001",
            actor="operator",
        )

    def test_inspection_preserves_documents_and_reports_uncertainty(self):
        receipt = receipt_path(self.path, "op-1")
        before = self.path.read_bytes(), receipt.read_bytes()
        report = inspect_operation(self.path, "op-1")
        self.assertEqual(report["receipt"]["state"], "pending")
        self.assertFalse(report["physical_outcome_verified"])
        self.assertFalse(report["historical_result_available"])
        self.assertTrue(report["budget"]["valid"])
        self.assertEqual(before, (self.path.read_bytes(), receipt.read_bytes()))

    def test_abandon_never_unlocks_budget_or_reexecutes(self):
        before = self.path.read_bytes()
        closed = self.close()
        self.assertEqual(closed["state"], "abandoned")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.invoke()["error"]["code"], "operation_abandoned")
        self.assertEqual(self.runtime.run.call_count, 1)
        audit = inspect_operation(self.path, "op-1")
        self.assertEqual(audit["receipt"]["resolution"]["actor"], "operator")
        self.assertFalse(audit["historical_result_available"])

    def test_stale_digest_and_second_close_rejected(self):
        before = receipt_path(self.path, "op-1").read_bytes()
        with self.assertRaises(ValueError):
            self.close("0" * 64)
        self.assertEqual(receipt_path(self.path, "op-1").read_bytes(), before)
        self.close()
        with self.assertRaises(ValueError):
            self.close()

    def test_completed_result_cannot_be_abandoned(self):
        path = receipt_path(self.path, "op-1")
        receipt = read_receipt(path)
        receipt.update(
            state="complete",
            result={
                "success": False,
                "message": "failed",
                "completed_steps": 0,
                "confirmation_reports": [],
            },
        )
        _write(path, receipt)
        with self.assertRaises(ValueError):
            self.close()
        self.assertTrue(inspect_operation(self.path, "op-1")["historical_result_available"])

    def test_bad_budget_does_not_hide_valid_receipt(self):
        self.path.write_text("{}")
        report = inspect_operation(self.path, "op-1")
        self.assertFalse(report["budget"]["valid"])
        self.assertEqual(report["receipt"]["state"], "pending")

    def test_corrupt_receipt_cannot_be_closed(self):
        receipt_path(self.path, "op-1").write_text("{}")
        with self.assertRaises(ValueError):
            self.close("0" * 64)
        with self.assertRaises(ValueError):
            inspect_operation(self.path, "op-1")

    def test_write_failure_leaves_pending(self):
        with (
            patch("embodied_agent.operation_admin._write", side_effect=OSError("disk")),
            self.assertRaises(OSError),
        ):
            self.close()
        self.assertEqual(read_receipt(receipt_path(self.path, "op-1"))["state"], "pending")

    def test_malformed_terminal_schema_rejected(self):
        self.close()
        path = receipt_path(self.path, "op-1")
        receipt = read_receipt(path)
        receipt["result"] = {"success": True}
        _write(path, receipt)
        with self.assertRaises(ValueError):
            inspect_operation(self.path, "op-1")
        self.assertIsNotNone(self.invoke()["error"])
        self.assertEqual(self.runtime.run.call_count, 1)

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "embodied_agent.operation_admin", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_cli_inspection_and_lock_conflict(self):
        args = ("inspect", "--checkpoint", str(self.path), "--operation-id", "op-1")
        child = self.cli(*args)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertTrue(json.loads(child.stdout)["ok"])
        with checkpoint_lock(self.path):
            child = self.cli(*args)
        self.assertEqual(child.returncode, 3)
        self.assertEqual(child.stdout, "")

    def test_cli_abandon_requires_review_digest(self):
        checksum = inspect_operation(self.path, "op-1")["receipt_sha256"]
        child = self.cli(
            "abandon",
            "--checkpoint",
            str(self.path),
            "--operation-id",
            "op-1",
            "--expected-receipt-sha256",
            checksum,
            "--reason",
            "cancelled",
            "--evidence-ref",
            "review-001",
            "--actor",
            "operator",
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout)["result"]["state"], "abandoned")
        self.assertEqual(self.invoke()["error"]["code"], "operation_abandoned")
