import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.validate_operational_lifecycle import validate


class LifecycleAcceptanceTests(unittest.TestCase):
    def test_real_subprocess_chain_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            report = validate(output)
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["stages"]), 15)
            self.assertTrue(
                all(r["exit_code"] == r["expected_exit_code"] for r in report["stages"])
            )
            for path, digest in report["artifact_sha256"].items():
                self.assertEqual(hashlib.sha256((output / path).read_bytes()).hexdigest(), digest)
            self.assertEqual(report, json.loads((output / "report.json").read_text()))
            self.assertTrue(any(".operations/" in path for path in report["artifact_sha256"]))

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "keep"
            marker.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                validate(directory)
            self.assertEqual(marker.read_text(), "unchanged")
            self.assertFalse((Path(directory) / "report.json").exists())

    def test_failed_stage_writes_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch(
                    "scripts.validate_operational_lifecycle.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 7, "", "injected failure"),
                ),
                self.assertRaises(AssertionError),
            ):
                validate(output)
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["stages"][0]["exit_code"], 7)
            self.assertEqual(report["failure"]["type"], "AssertionError")
