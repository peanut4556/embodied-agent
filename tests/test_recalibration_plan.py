import copy
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_budget_recalibration_cli as fixtures

from embodied_agent.budget_checkpoint import load_budget, save_budget
from embodied_agent.budget_recalibration import StaleRecalibration, inspect_recalibration
from embodied_agent.budget_recalibration_plan import apply_plan, preview_plan
from embodied_agent.checkpointed_runtime import checkpoint_lock


class RecalibrationPlanTests(unittest.TestCase):
    def setUp(self):
        fixtures.MaintenanceTests.setUp(self)
        pick = copy.deepcopy(self.data)
        pick["action"] = "pick"
        pick["calibration"]["batch_id"] = "pick-cal"
        pick["validation"]["batch_id"] = "pick-val"
        self.plan = self.path.parent / "plan.json"
        self.data_plan = {"schema": 1, "requests": [self.data, pick]}
        self.write()

    def write(self):
        self.plan.write_text(json.dumps(self.data_plan))

    def apply(self, preview):
        return apply_plan(
            self.path,
            self.plan,
            expected_checkpoint_sha256=preview["checkpoint_sha256"],
            expected_plan_sha256=preview["plan_sha256"],
        )

    def test_preview_and_single_publish_preserve_other_isolation(self):
        self.budget.degraded.add("place")
        save_budget(self.budget, self.path)
        before = self.path.read_bytes(), self.plan.read_bytes()
        preview = preview_plan(self.path, self.plan)
        self.assertEqual(before, (self.path.read_bytes(), self.plan.read_bytes()))
        self.assertEqual(preview["after_degraded"], ["place"])
        self.assertEqual(preview["after_revision"], 2)
        with patch(
            "embodied_agent.budget_recalibration_plan.save_budget", wraps=save_budget
        ) as save:
            self.assertTrue(self.apply(preview)["applied"])
        save.assert_called_once()
        budget = load_budget(self.path)
        self.assertEqual(budget.degraded, {"place"})
        self.assertEqual(
            [row["action"] for row in budget.recalibration_history()], ["verify", "pick"]
        )
        self.assertEqual(
            inspect_recalibration(self.path, self.request)["status"], "applied_current"
        )

    def test_second_validation_failure_never_publishes_first(self):
        preview = preview_plan(self.path, self.plan)
        self.data_plan["requests"][1]["validation"]["durations_s"] = [0.09] * 20
        self.write()
        before = self.path.read_bytes()
        with (
            patch("embodied_agent.budget_recalibration_plan.save_budget") as save,
            self.assertRaises(ValueError),
        ):
            self.apply(preview)
        save.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(load_budget(self.path).revision, 0)

    def test_duplicate_actions_and_cross_role_batch_reuse(self):
        original = copy.deepcopy(self.data_plan)
        for field in ("action", "batch"):
            self.data_plan = copy.deepcopy(original)
            second = self.data_plan["requests"][1]
            if field == "action":
                second["action"] = "verify"
            else:
                second["validation"]["batch_id"] = "cal-1"
            self.write()
            with self.assertRaises(ValueError):
                preview_plan(self.path, self.plan)

    def test_plan_bounds_and_strict_json(self):
        for value in (
            {"schema": True, "requests": []},
            {"schema": 1, "requests": []},
            {"schema": 1, "requests": [self.data] * 17},
            {"schema": 1, "requests": [self.data], "extra": 0},
        ):
            self.plan.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                preview_plan(self.path, self.plan)
        self.plan.write_text('{"schema":1,"schema":1,"requests":[]}')
        with self.assertRaises(ValueError):
            preview_plan(self.path, self.plan)

    def test_changed_checkpoint_and_changed_plan_rejected(self):
        preview = preview_plan(self.path, self.plan)
        self.data_plan["requests"].reverse()
        self.write()
        with self.assertRaises(StaleRecalibration):
            self.apply(preview)
        self.data_plan["requests"].reverse()
        self.write()
        self.budget.degraded.add("place")
        save_budget(self.budget, self.path)
        before = self.path.read_bytes()
        with self.assertRaises(StaleRecalibration):
            self.apply(preview)
        self.assertEqual(before, self.path.read_bytes())

    def test_save_failure_keeps_entire_original(self):
        preview = preview_plan(self.path, self.plan)
        before = self.path.read_bytes()
        with (
            patch("embodied_agent.budget_recalibration_plan.save_budget", side_effect=OSError),
            self.assertRaises(OSError),
        ):
            self.apply(preview)
        self.assertEqual(before, self.path.read_bytes())

    def test_repeat_apply_and_prior_batch_reuse(self):
        preview = preview_plan(self.path, self.plan)
        self.apply(preview)
        with self.assertRaises(StaleRecalibration):
            self.apply(preview)
        budget = load_budget(self.path)
        budget.degraded.update(("pick", "verify"))
        save_budget(budget, self.path)
        with self.assertRaises(ValueError):
            preview_plan(self.path, self.plan)

    def test_target_must_be_degraded_and_policy_quarantine(self):
        for policy, degraded in (("fallback", {"pick", "verify"}), ("quarantine", {"verify"})):
            self.budget.drift_policy = policy
            self.budget.degraded = degraded
            save_budget(self.budget, self.path)
            with self.assertRaises(ValueError):
                preview_plan(self.path, self.plan)

    def cli(self, *args):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "embodied_agent.budget_recalibration_plan",
                *args,
                "--checkpoint",
                str(self.path),
                "--plan",
                str(self.plan),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_cli_roundtrip_and_repeated_apply(self):
        child = self.cli("preview")
        self.assertEqual(child.returncode, 0, child.stderr)
        preview = json.loads(child.stdout)["result"]
        args = (
            "apply",
            "--expected-checkpoint-sha256",
            preview["checkpoint_sha256"],
            "--expected-plan-sha256",
            preview["plan_sha256"],
        )
        child = self.cli(*args)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout)["result"]["after_revision"], 2)
        self.assertEqual(self.cli(*args).returncode, 3)

    def test_cli_lock_and_missing_digests(self):
        with checkpoint_lock(self.path):
            child = self.cli("preview")
        self.assertEqual(child.returncode, 3)
        self.assertEqual(json.loads(child.stderr)["cause_type"], "CheckpointBusy")
        self.assertEqual(child.stdout, "")
        self.assertEqual(self.cli("apply").returncode, 2)

    def test_process_exit_before_and_after_publish(self):
        code = """import os, sys
import embodied_agent.budget_recalibration_plan as module
preview = module.preview_plan(sys.argv[1], sys.argv[2])
original = module.save_budget
def save(*args):
    if sys.argv[3] == 'after': original(*args)
    os._exit(23)
module.save_budget = save
module.apply_plan(sys.argv[1], sys.argv[2],
    expected_checkpoint_sha256=preview['checkpoint_sha256'],
    expected_plan_sha256=preview['plan_sha256'])
"""
        before = self.path.read_bytes()
        for boundary in ("before", "after"):
            child = subprocess.run(
                [sys.executable, "-c", code, str(self.path), str(self.plan), boundary],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(child.returncode, 23, child.stderr)
            self.assertEqual(child.stdout, "")
            budget = load_budget(self.path)
            if boundary == "before":
                self.assertEqual(before, self.path.read_bytes())
                self.assertEqual(budget.revision, 0)
                self.assertEqual(budget.degraded, {"pick", "verify"})
            else:
                self.assertEqual(budget.revision, 2)
                self.assertEqual(budget.degraded, set())
                self.assertEqual(
                    inspect_recalibration(self.path, self.request)["status"], "applied_current"
                )
