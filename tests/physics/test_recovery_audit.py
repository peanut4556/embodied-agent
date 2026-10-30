import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embodied_agent.recovery_audit import audit_files, validate_report
from scripts.check_recovery_reports import scenarios


class RecoveryAuditTests(unittest.TestCase):
    def setUp(self):
        self.bundle = scenarios()
        self.confirmed = self.bundle["cases"]["confirmed"]["after_close"]

    def test_existing_scenarios_count_each_task_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text(json.dumps(self.bundle))
            audit = audit_files([path])
        self.assertEqual(audit["accepted_count"], 8)
        self.assertEqual(audit["rejected_count"], 0)
        self.assertEqual(audit["counts"]["observation_restored"], 3)
        self.assertEqual(audit["counts"]["visual_confirmation_succeeded"], 2)
        self.assertEqual(audit["counts"]["cleanup_required"], 2)
        self.assertEqual(audit["counts"]["restart_count"], 7)
        self.assertEqual(audit["cleanup_counts"], {"succeeded": 6, "failed": 2})

    def test_contradictory_flags_counts_and_versions_rejected(self):
        mutations = [
            ("diagnosis", "visual_confirmation_succeeded", False),
            ("diagnosis", "observation_restored", False),
            ("diagnosis", "cleanup_required", True),
            ("diagnosis", "terminal", 1),
            ("diagnosis", "restored_attempts", [True]),
            ("result", "restart_count", True),
            ("result", "max_restarts", 0),
            ("result", "schema_version", 3),
            ("result", "deadline", float("inf")),
            ("camera", "status", "pending"),
        ]
        for section, key, value in mutations:
            with self.subTest(key=key):
                report = copy.deepcopy(self.confirmed)
                report[section][key] = value
                with self.assertRaises(ValueError):
                    validate_report(report)

    def test_attempt_identity_and_time_are_checked(self):
        for key, value in (
            ("attempt", 2),
            ("new_session", "original"),
            ("completed_at", 4),
            ("task_time", 4),
            ("fault", "idle"),
        ):
            report = copy.deepcopy(self.confirmed)
            report["result"]["recovery_attempts"][0][key] = value
            with self.assertRaises(ValueError):
                validate_report(report)

    def test_bad_case_excluded_without_losing_valid_cases(self):
        self.bundle["cases"]["confirmed"]["after_close"]["diagnosis"]["terminal"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text(json.dumps(self.bundle))
            audit = audit_files([path])
        self.assertEqual(audit["accepted_count"], 7)
        self.assertEqual(audit["rejected_count"], 1)
        self.assertEqual(audit["counts"]["visual_confirmation_succeeded"], 1)
        self.assertEqual(audit["rejected"][0]["case"], "confirmed")

    def test_invalid_json_duplicate_keys_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, text in enumerate(("[]", "{", '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}')):
                path = Path(directory) / f"bad{index}.json"
                path.write_text(text)
                report = audit_files([path])
                self.assertEqual(report["accepted_count"], 0)
                self.assertEqual(report["rejected_count"], 1)

    def test_missing_file_duplicate_path_and_empty_input(self):
        self.assertEqual(audit_files([])["accepted_count"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single.json"
            path.write_text(json.dumps(self.confirmed))
            report = audit_files([path, path, Path(directory) / "missing.json"])
        self.assertEqual(report["accepted_count"], 1)
        self.assertEqual(report["rejected_count"], 2)

    def test_close_transition_and_cleanup_exception_must_match(self):
        for field in ("before_close", "cleanup_exception"):
            bundle = copy.deepcopy(self.bundle)
            case = bundle["cases"]["confirmed"]
            case[field] = (
                bundle["cases"]["restored_pending"]["before_close"]
                if field == "before_close"
                else "hidden failure"
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bundle.json"
                path.write_text(json.dumps(bundle))
                report = audit_files([path])
            self.assertEqual(report["rejected_count"], 1)

    def test_cli_nonzero_on_rejection_and_never_overwrites(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "bad.json", Path(directory) / "audit.json"
            path.write_text("{}")
            command = [
                sys.executable,
                str(root / "scripts/audit_recovery_reports.py"),
                str(path),
                "--output",
                str(output),
            ]
            run = subprocess.run(command, capture_output=True, text=True, cwd=root, check=False)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(output.read_text())["rejected_count"], 1)
            original = output.read_bytes()
            run = subprocess.run(command, capture_output=True, text=True, cwd=root, check=False)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(path.read_text(), "{}")
