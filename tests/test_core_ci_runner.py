import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_core_ci import load_suite, run_suite


class CoreRunnerTests(unittest.TestCase):
    def test_outcome_categories_and_subtest(self):
        class Cases(unittest.TestCase):
            def test_fail(self):
                with self.subTest(value=1):
                    self.fail("injected")

            def test_error(self):
                raise RuntimeError("injected")

            @unittest.skip("fixture")
            def test_skip(self):
                pass

            @unittest.expectedFailure
            def test_expected(self):
                self.fail("expected")

            @unittest.expectedFailure
            def test_unexpected(self):
                pass

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            path = Path(directory) / "results"
            report = run_suite(unittest.defaultTestLoader.loadTestsFromTestCase(Cases), path)
            self.assertFalse(report["successful"])
            self.assertEqual(report["tests_run"], 5)
            for key in (
                "failures",
                "errors",
                "skipped",
                "expected_failures",
                "unexpected_successes",
            ):
                self.assertEqual(len(report[key]), 1)
            self.assertEqual(report, json.loads((path / "tests.json").read_text()))
            self.assertIn("FAIL", (path / "summary.md").read_text())
            with self.assertRaises(FileExistsError):
                run_suite(unittest.TestSuite(), path)

    def test_manifest_rejects_empty_duplicates_and_missing_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest"
            for value in ("", "test_runtime test_runtime", "../unsafe", "test_missing_ci_fixture"):
                path.write_text(value)
                with self.assertRaises(ValueError):
                    load_suite(path)

    def test_success_report(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            suite = unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])
            report = run_suite(suite, Path(directory) / "result")
            self.assertTrue(report["successful"])
            self.assertEqual(report["tests_run"], 1)

    def test_missing_manifest_records_load_failure(self):
        from scripts.run_core_ci import run_manifest

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            self.assertEqual(run_manifest(Path(directory) / "missing", output), 3)
            report = json.loads((output / "runner-error.json").read_text())
            self.assertEqual(report["phase"], "load")
            self.assertEqual(report["cause_type"], "FileNotFoundError")
            self.assertFalse((output / "tests.json").exists())

    def test_empty_module_rejected_even_with_other_tests(self):
        import sys
        import types
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest"
            path.write_text("test_runtime test_empty_fixture")
            with (
                patch.dict(
                    sys.modules, {"test_empty_fixture": types.ModuleType("test_empty_fixture")}
                ),
                self.assertRaisesRegex(ValueError, "contains no test cases"),
            ):
                load_suite(path)

    def test_runner_exception_preserves_error_report(self):
        from unittest.mock import patch

        from scripts.run_core_ci import run_manifest

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            with patch("scripts.run_core_ci.load_suite", side_effect=RuntimeError("fixture")):
                self.assertEqual(run_manifest("unused", output), 3)
            self.assertIn("fixture", (output / "runner-error.json").read_text())
            before = (output / "runner-error.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run_manifest("unused", output)
            self.assertEqual(before, (output / "runner-error.json").read_bytes())

    def test_cli_import_failure_exit_and_evidence(self):
        import os
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest").write_text("test_missing_ci_fixture")
            child = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_core_ci.py",
                    "--manifest",
                    str(path / "manifest"),
                    "--output",
                    str(path / "result"),
                ],
                env=dict(os.environ),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(child.returncode, 3, child.stderr)
            self.assertIn(
                "test_missing_ci_fixture", (path / "result/runner-error.json").read_text()
            )

    def test_oversized_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest"
            path.write_text(" " * 65537)
            with self.assertRaises(ValueError):
                load_suite(path)
