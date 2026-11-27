import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.verify_wheel import verify


class WheelVerificationTests(unittest.TestCase):
    def test_missing_wheel_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with self.assertRaises(ValueError):
                verify(Path(directory) / "missing.whl", output)
            self.assertFalse(output.exists())

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "test.whl"
            wheel.write_bytes(b"fixture")
            with self.assertRaises(FileExistsError):
                verify(wheel, directory)
            self.assertEqual(wheel.read_bytes(), b"fixture")

    def test_install_failure_preserves_report(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "test.whl"
            wheel.write_bytes(b"fixture")
            output = Path(directory) / "output"
            with (
                patch("scripts.verify_wheel.venv.EnvBuilder.create"),
                patch(
                    "scripts.verify_wheel.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 1, "", "invalid wheel"),
                ),
                self.assertRaises(RuntimeError),
            ):
                verify(wheel, output)
            report = json.loads((output / "wheel-verification.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["steps"][0]["name"], "install")
            self.assertEqual(report["steps"][0]["stderr"], "invalid wheel")
