import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.validate_crash_boundaries import validate


class CrashBoundaryTests(unittest.TestCase):
    def test_real_process_matrix_and_committed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            report = validate(output)
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["cases"]), 7)
            rows = {r["phase"]: r for r in report["cases"]}
            self.assertEqual(rows["before_registration"]["receipt_before_retry"], "absent")
            self.assertEqual(rows["after_receipt_commit"]["receipt_before_retry"], "complete")
            for phase, row in rows.items():
                self.assertEqual(row["crash_exit"], 23)
                self.assertLessEqual(row["effects_after_retry"], 1)
                if phase not in ("before_registration", "after_receipt_commit"):
                    self.assertEqual(row["receipt_before_retry"], "pending")
                if phase in ("after_prepare", "after_action"):
                    self.assertIn("verify", row["degraded_before_retry"])
            for name, digest in report["artifact_sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(report, json.loads((output / "report.json").read_text()))

    def test_output_directory_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "keep"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                validate(directory)
            self.assertEqual(marker.read_text(), "keep")

    def test_incorrect_crash_exit_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch(
                    "scripts.validate_crash_boundaries.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ),
                self.assertRaises(AssertionError),
            ):
                validate(output)
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["cases"][0]["crash_exit"], 0)
