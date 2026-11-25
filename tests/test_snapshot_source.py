import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import canonical, save_budget
from embodied_agent.operation_snapshot import export_snapshot, read_snapshot, verify_snapshot
from embodied_agent.snapshot_diff import compare_snapshots


class SnapshotSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "budget.json"
        self.budget = ActionBudget({}, drift_policy="quarantine")
        save_budget(self.budget, self.path)

    def snap(self, name, source=None, sequence=None):
        path = self.root / (name + ".json")
        summary = export_snapshot(self.path, path, source_id=source, capture_sequence=sequence)
        self.assertEqual(summary, verify_snapshot(path))
        return path

    def test_source_metadata_roundtrip(self):
        path = self.snap("a", "robot-a", 1)
        body = read_snapshot(path)["snapshot"]
        self.assertEqual(body["schema"], 2)
        self.assertEqual(body["source_id"], "robot-a")
        self.assertEqual(body["capture_sequence"], 1)

    def test_matching_sources_compare_without_authenticity_claim(self):
        report = compare_snapshots(self.snap("a", "robot-a", 1), self.snap("b", "robot-a", 2))
        self.assertTrue(report["comparison_performed"])
        self.assertFalse(report["needs_review"])
        self.assertFalse(report["same_source_verified"])
        self.assertEqual(report["source"]["relation"], "matched_declared")
        self.assertEqual(report["source"]["order_relation"], "increasing_declared")

    def test_different_sources_gate_before_regression_analysis(self):
        report = compare_snapshots(self.snap("a", "robot-a"), self.snap("b", "robot-b"))
        self.assertFalse(report["comparison_performed"])
        self.assertIsNone(report["counts"])
        self.assertEqual(report["budget_flags"], [])
        self.assertIn("source_identity_mismatch", report["source"]["flags"])

    def test_legacy_remains_valid_but_default_comparison_is_gated(self):
        old = self.snap("old")
        before = old.read_bytes()
        self.assertEqual(read_snapshot(old)["snapshot"]["schema"], 1)
        report = compare_snapshots(old, old)
        self.assertFalse(report["comparison_performed"])
        self.assertIn("source_identity_unavailable", report["source"]["flags"])
        self.assertEqual(old.read_bytes(), before)

    def test_mixed_schema_comparison_needs_explicit_override(self):
        old, new = self.snap("old"), self.snap("new", "robot-a")
        self.assertFalse(compare_snapshots(old, new)["comparison_performed"])
        report = compare_snapshots(old, new, allow_source_override=True)
        self.assertTrue(report["comparison_performed"])
        self.assertTrue(report["needs_review"])
        self.assertTrue(report["source"]["override"])

    def test_reverse_sequence_blocks_comparison(self):
        report = compare_snapshots(self.snap("a", "robot-a", 2), self.snap("b", "robot-a", 1))
        self.assertFalse(report["comparison_performed"])
        self.assertIn("capture_order_reversed", report["source"]["flags"])

    def test_reused_sequence_with_changed_content_is_flagged(self):
        before = self.snap("a", "robot-a", 1)
        self.budget.degraded.add("pick")
        save_budget(self.budget, self.path)
        report = compare_snapshots(before, self.snap("b", "robot-a", 1))
        self.assertIn("capture_sequence_reused", report["source"]["flags"])
        self.assertFalse(report["comparison_performed"])

    def test_exact_same_snapshot_not_treated_as_sequence_reuse(self):
        path = self.snap("a", "robot-a", 1)
        self.assertFalse(compare_snapshots(path, path)["needs_review"])

    def test_invalid_metadata_never_publishes(self):
        for source, sequence in (
            ("", None),
            (" spaced ", None),
            (None, 1),
            ("a", True),
            ("a", 0),
            ("a", 1.5),
        ):
            output = self.root / "bad.json"
            with self.assertRaises(ValueError):
                export_snapshot(self.path, output, source_id=source, capture_sequence=sequence)
            self.assertFalse(output.exists())

    def test_rehashed_malformed_metadata_rejected(self):
        path = self.snap("a", "robot-a", 1)
        value = json.loads(path.read_text())
        value["snapshot"]["capture_sequence"] = True
        value["sha256"] = hashlib.sha256(canonical(value["snapshot"]).encode()).hexdigest()
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            verify_snapshot(path)

    def test_cli_metadata_and_override_status(self):
        output = self.root / "cli.json"
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "embodied_agent.operation_admin",
                "snapshot",
                "--checkpoint",
                str(self.path),
                "--output",
                str(output),
                "--source-id",
                "robot-a",
                "--capture-sequence",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout)["result"]["source_id"], "robot-a")
        old = self.snap("old")
        args = [
            sys.executable,
            "-m",
            "embodied_agent.operation_admin",
            "compare-snapshots",
            "--before",
            str(old),
            "--after",
            str(output),
        ]
        for override in (False, True):
            child = subprocess.run(
                args + (["--allow-source-override"] if override else []),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(child.returncode, 1, child.stderr)
            self.assertEqual(json.loads(child.stdout)["result"]["comparison_performed"], override)

    def test_override_parameter_is_strict_boolean(self):
        with self.assertRaises(TypeError):
            compare_snapshots("missing", "missing", allow_source_override="yes")
