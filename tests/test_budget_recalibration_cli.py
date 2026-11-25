import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import load_budget, save_budget
from embodied_agent.budget_recalibration import (
    StaleRecalibration,
    apply_recalibration,
    preview_recalibration,
)
from embodied_agent.checkpointed_runtime import checkpoint_lock


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "budget.json"
        self.request = Path(self.directory.name) / "request.json"
        self.budget = ActionBudget(
            {"pick": [0.01] * 20, "verify": [0.01] * 20}, drift_policy="quarantine"
        )
        self.budget.record("pick", 0.04, confirmed=True)
        self.budget.record("verify", 0.04, confirmed=True)
        save_budget(self.budget, self.path)
        self.data = {
            "schema": 1,
            "action": "verify",
            "calibration": {"batch_id": "cal-1", "durations_s": [0.03] * 20},
            "validation": {"batch_id": "val-1", "durations_s": [0.032] * 20},
        }
        self.request.write_text(json.dumps(self.data))

    def apply(self, preview):
        return apply_recalibration(
            self.path,
            self.request,
            expected_checkpoint_sha256=preview["checkpoint_sha256"],
            expected_request_sha256=preview["request_sha256"],
        )

    def test_preview_readonly_apply_changes_only_target(self):
        before = self.path.read_bytes(), self.request.read_bytes()
        preview = preview_recalibration(self.path, self.request)
        self.assertFalse(preview["applied"])
        self.assertEqual(before, (self.path.read_bytes(), self.request.read_bytes()))
        self.assertEqual(preview["after_degraded"], ["pick"])
        self.assertTrue(self.apply(preview)["applied"])
        budget = load_budget(self.path)
        self.assertTrue(budget.blocked("pick"))
        self.assertFalse(budget.blocked("verify"))
        self.assertEqual(budget.revision, 1)
        self.assertEqual(budget.recalibration_history()[0]["validation_batch_id"], "val-1")

    def test_checkpoint_change_rejects_old_preview(self):
        preview = preview_recalibration(self.path, self.request)
        self.budget.degraded.add("locate")
        save_budget(self.budget, self.path)
        before = self.path.read_bytes()
        with self.assertRaises(StaleRecalibration):
            self.apply(preview)
        self.assertEqual(self.path.read_bytes(), before)

    def test_request_change_rejects_old_preview(self):
        preview = preview_recalibration(self.path, self.request)
        self.data["calibration"]["batch_id"] = "cal-2"
        self.request.write_text(json.dumps(self.data))
        before = self.path.read_bytes()
        with self.assertRaises(StaleRecalibration):
            self.apply(preview)
        self.assertEqual(self.path.read_bytes(), before)

    def test_failed_validation_keeps_quarantine(self):
        self.data["validation"]["durations_s"] = [0.05] * 20
        self.request.write_text(json.dumps(self.data))
        with self.assertRaises(ValueError):
            preview_recalibration(self.path, self.request)
        self.assertTrue(load_budget(self.path).blocked("verify"))

    def test_duplicate_fields_and_unknown_schema_rejected(self):
        for raw in ('{"schema":1,"schema":1}', json.dumps(dict(self.data, schema=2))):
            self.request.write_text(raw)
            with self.assertRaises(ValueError):
                preview_recalibration(self.path, self.request)

    def test_input_bounds_and_nonfinite_values(self):
        for values in ([0.03] * 4097, [0.03] * 19, [float("nan")] * 20, "not-a-list"):
            self.data["calibration"]["durations_s"] = values
            self.request.write_text(json.dumps(self.data))
            with self.assertRaises(ValueError):
                preview_recalibration(self.path, self.request)

    def test_save_failure_does_not_change_original(self):
        preview = preview_recalibration(self.path, self.request)
        before = self.path.read_bytes()
        with (
            patch("embodied_agent.budget_recalibration.save_budget", side_effect=OSError("disk")),
            self.assertRaises(OSError),
        ):
            self.apply(preview)
        self.assertEqual(self.path.read_bytes(), before)

    def test_repeat_apply_stale_and_batch_reuse_rejected(self):
        preview = preview_recalibration(self.path, self.request)
        self.apply(preview)
        with self.assertRaises(StaleRecalibration):
            self.apply(preview)
        budget = load_budget(self.path)
        budget.record("verify", 0.06, confirmed=True)
        save_budget(budget, self.path)
        with self.assertRaises(ValueError):
            preview_recalibration(self.path, self.request)

    def cli(self, *args):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "embodied_agent.budget_recalibration",
                *args,
                "--checkpoint",
                str(self.path),
                "--request",
                str(self.request),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_cli_preview_apply_and_stale_exit(self):
        child = self.cli("preview")
        self.assertEqual(child.returncode, 0, child.stderr)
        preview = json.loads(child.stdout)["result"]
        args = (
            "apply",
            "--expected-checkpoint-sha256",
            preview["checkpoint_sha256"],
            "--expected-request-sha256",
            preview["request_sha256"],
        )
        self.assertEqual(self.cli(*args).returncode, 0)
        child = self.cli(*args)
        self.assertEqual(child.returncode, 3)
        self.assertEqual(json.loads(child.stderr)["cause_type"], "StaleRecalibration")
        self.assertEqual(child.stdout, "")

    def test_cli_lock_conflict_and_missing_approval_digests(self):
        with checkpoint_lock(self.path):
            child = self.cli("preview")
        self.assertEqual(child.returncode, 3)
        self.assertEqual(json.loads(child.stderr)["cause_type"], "CheckpointBusy")
        self.assertEqual(self.cli("apply").returncode, 2)
