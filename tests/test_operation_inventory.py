import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import save_budget
from embodied_agent.checkpointed_runtime import checkpoint_lock
from embodied_agent.operation_inventory import inspect_receipt_key, list_operations
from embodied_agent.operation_receipt import _write


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "budget.json"
        save_budget(ActionBudget({}, drift_policy="quarantine"), self.path)
        self.receipts = self.path.with_name(self.path.name + ".operations")
        self.receipts.mkdir()
        self.pending = {"schema": 1, "request_sha256": "a" * 64, "state": "pending", "result": None}

    def write(self, number, state="pending"):
        key = f"{number:064x}"
        receipt = dict(self.pending)
        if state == "complete":
            receipt.update(
                state="complete",
                result={
                    "success": False,
                    "message": "failed",
                    "completed_steps": 0,
                    "confirmation_reports": [],
                },
            )
        elif state == "abandoned":
            receipt.update(
                schema=2,
                state="abandoned",
                resolution={
                    "reason": "cancelled",
                    "evidence_ref": "review",
                    "actor": "operator",
                    "previous_receipt_sha256": "b" * 64,
                },
            )
        _write(self.receipts / (key + ".json"), receipt)
        return key

    def test_mixed_page_preserves_invalid_records(self):
        self.write(1)
        self.write(2, "complete")
        self.write(3, "abandoned")
        (self.receipts / ("4".zfill(64) + ".json")).write_text("{}")
        report = list_operations(self.path)
        self.assertEqual(
            report["page_counts"], {"pending": 1, "complete": 1, "abandoned": 1, "invalid": 1}
        )
        self.assertTrue(report["has_errors"])
        self.assertFalse(report["items"][1]["success"])
        self.assertEqual(report["matched_entries"], 4)

    def test_pagination_no_duplicates_and_page_local_counts(self):
        keys = [self.write(i) for i in range(1, 6)]
        seen, cursor = [], None
        for expected_size in (2, 2, 1):
            report = list_operations(self.path, after=cursor, limit=2)
            self.assertEqual(report["page_counts"]["pending"], expected_size)
            seen.extend(row["receipt_key"] for row in report["items"])
            cursor = report["next_cursor"]
        self.assertEqual(seen, keys)
        self.assertIsNone(cursor)

    def test_empty_and_missing_directory(self):
        for exists in (True, False):
            if not exists:
                self.receipts.rmdir()
            report = list_operations(self.path)
            self.assertEqual(report["items"], [])
            self.assertFalse(report["has_more"])
            self.assertFalse(report["has_errors"])

    def test_invalid_cursor_and_page_size(self):
        for value in (True, 0, 26, 1.5):
            with self.assertRaises(ValueError):
                list_operations(self.path, limit=value)
        with self.assertRaises(ValueError):
            list_operations(self.path, after="../escape")
        with self.assertRaises(ValueError):
            inspect_receipt_key(self.path, "../escape")

    def test_scan_limit_fails_instead_of_truncating(self):
        for i in range(3):
            self.write(i)
        with (
            patch("embodied_agent.operation_inventory.MAX_ENTRIES", 2),
            self.assertRaises(ValueError),
        ):
            list_operations(self.path, limit=1)

    def test_unrelated_and_symlink_entries_are_counted_not_read(self):
        key = self.write(1)
        (self.receipts / "notes.txt").write_text("notes")
        (self.receipts / ("2".zfill(64) + ".json")).symlink_to(self.receipts / (key + ".json"))
        report = list_operations(self.path)
        self.assertEqual(report["ignored_entries"], 2)
        self.assertEqual(len(report["items"]), 1)

    def test_bad_budget_does_not_hide_receipt_page(self):
        self.write(1)
        self.path.write_text("{}")
        report = list_operations(self.path)
        self.assertEqual(len(report["items"]), 1)
        self.assertFalse(report["budget"]["valid"])
        self.assertTrue(report["has_errors"])

    def test_only_selected_page_is_parsed(self):
        self.write(1)
        self.write(2)
        with patch(
            "embodied_agent.operation_inventory.read_receipt", return_value=self.pending
        ) as reader:
            list_operations(self.path, limit=1)
        self.assertEqual(reader.call_count, 1)

    def test_inspect_key_returns_detached_read_only_details(self):
        key = self.write(1)
        before = self.path.read_bytes(), (self.receipts / (key + ".json")).read_bytes()
        detail = inspect_receipt_key(self.path, key)
        self.assertEqual(detail["receipt"]["state"], "pending")
        self.assertFalse(detail["physical_outcome_verified"])
        self.assertEqual(
            before, (self.path.read_bytes(), (self.receipts / (key + ".json")).read_bytes())
        )

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "embodied_agent.operation_admin", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_cli_healthy_and_corrupt_page_exit_codes(self):
        key = self.write(1)
        args = ("list", "--checkpoint", str(self.path))
        self.assertEqual(self.cli(*args).returncode, 0)
        (self.receipts / (key + ".json")).write_text("{}")
        child = self.cli(*args)
        self.assertEqual(child.returncode, 1)
        self.assertTrue(json.loads(child.stdout)["result"]["has_errors"])

    def test_cli_lock_conflict_and_inspect_key(self):
        key = self.write(1)
        args = ("inspect-key", "--checkpoint", str(self.path), "--receipt-key", key)
        child = self.cli(*args)
        self.assertEqual(child.returncode, 0, child.stderr)
        with checkpoint_lock(self.path):
            child = self.cli(*args)
        self.assertEqual(child.returncode, 3)
        self.assertEqual(child.stdout, "")

    def test_cursor_beyond_end_returns_empty_page(self):
        self.write(1)
        report = list_operations(self.path, after="f" * 64)
        self.assertEqual(report["items"], [])
        self.assertIsNone(report["next_cursor"])
        self.assertEqual(report["matched_entries"], 1)
