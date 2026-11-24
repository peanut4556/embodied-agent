import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import canonical, save_budget
from embodied_agent.checkpointed_runtime import checkpoint_lock
from embodied_agent.operation_receipt import _write
from embodied_agent.operation_snapshot import export_snapshot, verify_snapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "budget.json"
        self.output = Path(self.directory.name) / "audit.json"
        self.receipts = self.path.with_name(self.path.name + ".operations")
        self.receipts.mkdir()
        self.file = self.receipts / ("a" * 64 + ".json")
        save_budget(ActionBudget({}, drift_policy="quarantine"), self.path)
        _write(
            self.file, {"schema": 1, "request_sha256": "b" * 64, "state": "pending", "result": None}
        )

    def test_full_export_and_offline_verification(self):
        before = self.path.read_bytes(), self.file.read_bytes()
        summary = export_snapshot(self.path, self.output)
        self.assertEqual(summary, verify_snapshot(self.output))
        self.assertEqual(summary["counts"]["pending"], 1)
        self.assertEqual(before, (self.path.read_bytes(), self.file.read_bytes()))
        self.file.unlink()
        self.path.unlink()
        self.assertEqual(summary, verify_snapshot(self.output))

    def test_corrupt_sources_preserved_as_invalid(self):
        self.file.write_text("{}")
        self.path.write_text("{}")
        summary = export_snapshot(self.path, self.output)
        self.assertTrue(summary["has_errors"])
        self.assertEqual(summary["counts"]["invalid"], 1)
        self.assertEqual(summary, verify_snapshot(self.output))

    def test_existing_destination_not_overwritten(self):
        self.output.write_text("keep")
        with self.assertRaises(FileExistsError):
            export_snapshot(self.path, self.output)
        self.assertEqual(self.output.read_text(), "keep")
        self.assertEqual(list(self.output.parent.glob(".snapshot-*")), [])

    def test_source_destination_and_receipt_directory_rejected(self):
        for output in (
            self.path,
            self.path.with_name(self.path.name + ".lock"),
            self.receipts / "new.json",
        ):
            with self.assertRaises(ValueError):
                export_snapshot(self.path, output)

    def test_receipt_count_limit_has_no_partial_output(self):
        with (
            patch("embodied_agent.operation_snapshot.MAX_RECEIPTS", 0),
            self.assertRaises(ValueError),
        ):
            export_snapshot(self.path, self.output)
        self.assertFalse(self.output.exists())

    def test_aggregate_budget_has_no_partial_output(self):
        with (
            patch("embodied_agent.operation_snapshot.MAX_SOURCE_BYTES", 10),
            self.assertRaises(ValueError),
        ):
            export_snapshot(self.path, self.output)
        self.assertFalse(self.output.exists())

    def test_oversized_single_source_marked_without_false_digest(self):
        self.file.write_text("x" * 1000)
        with patch("embodied_agent.operation_snapshot.LIMIT", 500):
            export_snapshot(self.path, self.output)
        item = json.loads(self.output.read_text())["snapshot"]["items"][0]
        self.assertIsNone(item["source_sha256"])
        self.assertEqual(item["error"], "SourceTooLarge")
        self.assertTrue(verify_snapshot(self.output)["has_errors"])

    def test_tampering_and_rehashed_inconsistent_counts_rejected(self):
        export_snapshot(self.path, self.output)
        data = json.loads(self.output.read_text())
        data["snapshot"]["counts"]["pending"] = 9
        self.output.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            verify_snapshot(self.output)
        data["sha256"] = hashlib.sha256(canonical(data["snapshot"]).encode()).hexdigest()
        self.output.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            verify_snapshot(self.output)

    def test_failed_publish_cleans_temporary(self):
        with (
            patch("embodied_agent.operation_snapshot.os.link", side_effect=OSError("disk")),
            self.assertRaises(OSError),
        ):
            export_snapshot(self.path, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".snapshot-*")), [])

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "embodied_agent.operation_admin", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_cli_export_verify_and_busy(self):
        args = ("snapshot", "--checkpoint", str(self.path), "--output", str(self.output))
        with checkpoint_lock(self.path):
            child = self.cli(*args)
        self.assertEqual(child.returncode, 3)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.cli(*args).returncode, 0)
        self.assertEqual(self.cli("verify-snapshot", "--snapshot", str(self.output)).returncode, 0)

    def test_cli_invalid_sources_exit_one_with_artifact(self):
        self.file.write_text("{}")
        child = self.cli("snapshot", "--checkpoint", str(self.path), "--output", str(self.output))
        self.assertEqual(child.returncode, 1, child.stderr)
        self.assertTrue(self.output.exists())
        self.assertEqual(self.cli("verify-snapshot", "--snapshot", str(self.output)).returncode, 1)

    def test_empty_directory_snapshot(self):
        self.file.unlink()
        self.receipts.rmdir()
        self.assertEqual(export_snapshot(self.path, self.output)["counts"]["pending"], 0)
        self.assertFalse(verify_snapshot(self.output)["has_errors"])
