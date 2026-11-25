import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_recalibration_plan as fixtures

from embodied_agent.budget_checkpoint import load_budget, save_budget
from embodied_agent.budget_recalibration import apply_recalibration, preview_recalibration
from embodied_agent.budget_recalibration_plan import inspect_plan, preview_plan
from embodied_agent.checkpointed_runtime import checkpoint_lock


class PlanStatusTests(unittest.TestCase):
    setUp = fixtures.RecalibrationPlanTests.setUp
    write = fixtures.RecalibrationPlanTests.write
    apply = fixtures.RecalibrationPlanTests.apply
    cli = fixtures.RecalibrationPlanTests.cli

    def status(self):
        before = self.path.read_bytes(), self.plan.read_bytes()
        with patch(
            "embodied_agent.budget_recalibration_plan.load_budget", wraps=load_budget
        ) as load:
            result = inspect_plan(self.path, self.plan)
        load.assert_called_once()
        self.assertEqual(before, (self.path.read_bytes(), self.plan.read_bytes()))
        self.assertFalse(result["atomic_application_verified"])
        self.assertEqual(result["evidence_scope"], "current_checkpoint_history")
        return result

    def test_not_found_and_all_matched(self):
        self.assertEqual(self.status()["status"], "not_found")
        self.apply(preview_plan(self.path, self.plan))
        result = self.status()
        self.assertEqual(result["status"], "all_matched")
        self.assertEqual(result["counts"]["applied_current"], 2)
        self.assertEqual(result["matched_count"], 2)
        self.assertEqual(result["currently_blocked_actions"], [])
        self.assertTrue(all("request_sha256" not in r for r in result["results"]))

    def test_partial_is_not_plan_commit_failure_claim(self):
        preview = preview_recalibration(self.path, self.request)
        apply_recalibration(
            self.path,
            self.request,
            expected_checkpoint_sha256=preview["checkpoint_sha256"],
            expected_request_sha256=preview["request_sha256"],
        )
        result = self.status()
        self.assertEqual(result["status"], "partially_matched")
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["counts"]["not_found"], 1)
        self.assertEqual(result["currently_blocked_actions"], ["pick"])

    def test_conflict_takes_precedence_over_partial(self):
        self.apply(preview_plan(self.path, self.plan))
        self.data_plan["requests"][1]["validation"]["durations_s"] = [0.031] * 20
        self.write()
        result = self.status()
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["counts"]["batch_conflict"], 1)

    def test_matched_history_does_not_imply_current_readiness(self):
        self.apply(preview_plan(self.path, self.plan))
        budget = load_budget(self.path)
        budget.record("verify", 0.1, confirmed=True)
        budget.record("pick", 0.1, confirmed=True)
        budget = budget.validated_recalibrated(
            "pick",
            [0.04] * 20,
            batch_id="new-cal",
            validation_samples=[0.041] * 20,
            validation_batch_id="new-val",
        )
        save_budget(budget, self.path)
        result = self.status()
        self.assertEqual(result["status"], "all_matched")
        self.assertEqual(result["counts"]["applied_then_degraded"], 1)
        self.assertEqual(result["counts"]["superseded"], 1)
        self.assertEqual(result["currently_blocked_actions"], ["verify"])

    def test_plan_order_and_format_do_not_change_matches(self):
        self.apply(preview_plan(self.path, self.plan))
        original = self.status()["plan_sha256"]
        self.data_plan["requests"].reverse()
        self.plan.write_text(json.dumps(self.data_plan, indent=2))
        result = self.status()
        self.assertEqual(result["status"], "all_matched")
        self.assertNotEqual(result["plan_sha256"], original)
        self.assertEqual([r["action"] for r in result["results"]], ["pick", "verify"])

    def test_cli_success_conflict_and_lock(self):
        child = self.cli("status")
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout)["result"]["status"], "not_found")
        with checkpoint_lock(self.path):
            child = self.cli("status")
        self.assertEqual(child.returncode, 3)
        self.assertEqual(json.loads(child.stderr)["cause_type"], "CheckpointBusy")
        self.assertEqual(child.stdout, "")

    def test_bad_plan_and_checkpoint_are_not_absent_history(self):
        self.plan.write_text('{"schema":1,"schema":1}')
        self.assertEqual(self.cli("status").returncode, 3)
        self.write()
        self.path.write_text("{}")
        child = self.cli("status")
        self.assertEqual(child.returncode, 3)
        self.assertEqual(child.stdout, "")

    def test_batch_saved_without_response_then_cli_status(self):
        code = """import os, sys
from embodied_agent.budget_recalibration_plan import preview_plan, apply_plan
p = preview_plan(sys.argv[1], sys.argv[2])
apply_plan(sys.argv[1], sys.argv[2], expected_checkpoint_sha256=p['checkpoint_sha256'],
           expected_plan_sha256=p['plan_sha256'])
os._exit(23)
"""
        child = subprocess.run(
            [sys.executable, "-c", code, str(self.path), str(self.plan)],
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
        self.assertEqual(result["status"], "all_matched")
        self.assertEqual(result["matched_count"], 2)
        self.assertFalse(result["atomic_application_verified"])
        self.assertEqual(self.path.read_bytes(), before)
