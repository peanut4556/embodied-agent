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
