import json
import subprocess
import sys
import unittest

import test_budget_recalibration_cli as fixtures

from embodied_agent.budget_checkpoint import load_budget, save_budget
from embodied_agent.budget_recalibration import inspect_recalibration, preview_recalibration
from embodied_agent.checkpointed_runtime import checkpoint_lock


class RecalibrationStatusTests(unittest.TestCase):
    setUp = fixtures.MaintenanceTests.setUp
    apply = fixtures.MaintenanceTests.apply
    cli = fixtures.MaintenanceTests.cli

    def status(self):
        before = self.path.read_bytes(), self.request.read_bytes()
        result = inspect_recalibration(self.path, self.request)
        self.assertEqual(before, (self.path.read_bytes(), self.request.read_bytes()))
        return result

    def applied(self):
        self.apply(preview_recalibration(self.path, self.request))

    def test_not_found_does_not_claim_never_applied(self):
        result = self.status()
        self.assertEqual(result["status"], "not_found")
        self.assertIsNone(result["matched_revision"])
        self.assertEqual(result["evidence_scope"], "current_checkpoint_history")
        self.assertTrue(result["currently_blocked"])

    def test_current_and_other_action_update(self):
        self.applied()
        budget = load_budget(self.path).validated_recalibrated(
            "pick",
            [0.03] * 20,
            batch_id="pick-cal",
            validation_samples=[0.032] * 20,
            validation_batch_id="pick-val",
        )
        save_budget(budget, self.path)
        result = self.status()
        self.assertEqual(result["status"], "applied_current")
        self.assertEqual(result["current_revision"], 2)
        self.assertEqual(result["matched_revision"], 1)
        self.assertFalse(result["currently_blocked"])

    def test_degraded_after_apply(self):
        self.applied()
        budget = load_budget(self.path)
        budget.record("verify", 0.1, confirmed=True)
        save_budget(budget, self.path)
        self.assertEqual(self.status()["status"], "applied_then_degraded")
        self.assertTrue(self.status()["currently_blocked"])

    def test_superseded_and_current_degradation_are_separate(self):
        self.applied()
        budget = load_budget(self.path)
        budget.record("verify", 0.1, confirmed=True)
        budget = budget.validated_recalibrated(
            "verify",
            [0.04] * 20,
            batch_id="new-cal",
            validation_samples=[0.042] * 20,
            validation_batch_id="new-val",
        )
        budget.record("verify", 0.1, confirmed=True)
        save_budget(budget, self.path)
        result = self.status()
        self.assertEqual(result["status"], "superseded")
        self.assertEqual(result["later_action_revisions"], [2])
        self.assertTrue(result["currently_degraded"])

    def test_content_or_action_or_role_conflict(self):
        self.applied()
        original = json.loads(self.request.read_text())
        changes = [
            dict(original, action="pick"),
            dict(original, calibration=original["validation"], validation=original["calibration"]),
            dict(original, calibration={"batch_id": "cal-1", "durations_s": [0.031] * 20}),
        ]
        for data in changes:
            self.request.write_text(json.dumps(data))
            result = self.status()
            self.assertEqual(result["status"], "batch_conflict")
            self.assertEqual(result["conflicting_revisions"], [1])
            self.assertIsNone(result["matched_record"])

    def test_json_format_does_not_change_history_match(self):
        self.applied()
        self.request.write_text(json.dumps(self.data, indent=4, sort_keys=True))
        self.assertEqual(self.status()["status"], "applied_current")

    def test_no_validation_history_is_not_validated_match(self):
        budget = self.budget.recalibrated("verify", [0.03] * 20, batch_id="cal-1")
        save_budget(budget, self.path)
        self.assertEqual(self.status()["status"], "batch_conflict")

    def test_invalid_input_is_error_not_not_found(self):
        for values in ([True] * 20, [-1] * 20, [], [float("inf")] * 20):
            self.data["calibration"]["durations_s"] = values
            self.request.write_text(json.dumps(self.data))
            with self.assertRaises(ValueError):
                self.status()

    def test_cli_lock_and_corrupt_checkpoint(self):
        with checkpoint_lock(self.path):
            child = self.cli("status")
        self.assertEqual(child.returncode, 3)
        self.assertEqual(json.loads(child.stderr)["cause_type"], "CheckpointBusy")
        self.path.write_text("{}")
        child = self.cli("status")
        self.assertEqual(child.returncode, 3)
        self.assertEqual(child.stdout, "")

    def test_real_process_saved_then_lost_response(self):
        code = """import os, sys
from embodied_agent.budget_recalibration import preview_recalibration, apply_recalibration
p = preview_recalibration(sys.argv[1], sys.argv[2])
apply_recalibration(sys.argv[1], sys.argv[2],
    expected_checkpoint_sha256=p['checkpoint_sha256'],
    expected_request_sha256=p['request_sha256'])
os._exit(23)
"""
        child = subprocess.run(
            [sys.executable, "-c", code, str(self.path), str(self.request)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(child.returncode, 23, child.stderr)
        self.assertEqual(child.stdout, "")
        before = self.path.read_bytes()
        child = self.cli("status")
        self.assertEqual(child.returncode, 0, child.stderr)
        result = json.loads(child.stdout)["result"]
        self.assertEqual(result["status"], "applied_current")
        self.assertEqual(result["matched_revision"], 1)
        self.assertEqual(self.path.read_bytes(), before)
