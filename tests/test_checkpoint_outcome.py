import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import load_budget, save_budget
from embodied_agent.checkpointed_runtime import checkpoint_lock, run_checkpointed_report
from embodied_agent.models import TaskResult


class CheckpointOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state.json"
        save_budget(ActionBudget({}, drift_policy="quarantine"), self.path)
        self.runtime = Mock()
        self.runtime.run.return_value = TaskResult(True, "done", 1)

    def run_report(self):
        return run_checkpointed_report(self.runtime, "test", self.path, task_timeout_s=1)

    def test_success_and_task_failure_have_committed_state(self):
        for success in (True, False):
            self.runtime.run.return_value = TaskResult(success, "done", int(success))
            report = self.run_report()
            self.assertEqual(report["success"], success)
            self.assertIsNone(report["error"])
            self.assertEqual(report["checkpoint"]["checkpoint_state"], "committed")
            self.assertEqual(report["checkpoint"]["phase"], "complete")

    def test_missing_and_corrupt_file_never_start_runtime(self):
        for value in (None, "{}"):
            if value is None:
                self.path.unlink()
            else:
                self.path.write_text(value)
            report = self.run_report()
            self.assertEqual(report["error"]["code"], "checkpoint_load_failed")
            self.assertFalse(report["error"]["action_outcome_unknown"])
            self.assertIsNone(report["task_result"])
        self.runtime.run.assert_not_called()

    def test_prepare_failure_is_not_misreported_as_execution(self):
        with patch(
            "embodied_agent.checkpointed_runtime.save_budget",
            side_effect=OSError("secret device path"),
        ):
            report = self.run_report()
        self.assertEqual(report["error"]["code"], "checkpoint_prepare_failed")
        self.assertFalse(report["checkpoint"]["execution_started"])
        self.runtime.run.assert_not_called()
        self.assertNotIn("secret device", json.dumps(report))

    def test_execution_failure_marks_unknown_and_keeps_prepared_state(self):
        self.runtime.run.side_effect = RuntimeError("sensitive details")
        report = self.run_report()
        self.assertEqual(report["error"]["code"], "checkpoint_execute_failed")
        self.assertTrue(report["error"]["action_outcome_unknown"])
        self.assertEqual(report["checkpoint"]["checkpoint_state"], "prepared")
        self.assertTrue(load_budget(self.path).blocked("pick"))
        self.assertNotIn("sensitive details", json.dumps(report))

    def test_commit_failure_does_not_expose_success(self):
        count = 0

        def save(budget, path):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("disk")
            save_budget(budget, path)

        with patch("embodied_agent.checkpointed_runtime.save_budget", side_effect=save):
            report = self.run_report()
        self.assertFalse(report["success"])
        self.assertIsNone(report["task_result"])
        self.assertEqual(report["error"]["code"], "checkpoint_commit_failed")
        self.assertEqual(report["checkpoint"]["checkpoint_state"], "not_confirmed")
        self.assertTrue(report["error"]["action_outcome_unknown"])

    def test_interrupt_not_swallowed(self):
        self.runtime.run.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_report()
        self.assertTrue(load_budget(self.path).blocked("pick"))

    def cli(self, json_mode=True):
        args = [
            sys.executable,
            "-m",
            "embodied_agent",
            "--budget-checkpoint",
            str(self.path),
            "--task-timeout",
            "1",
        ]
        if json_mode:
            args.append("--json")
        return subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)

    def test_cli_busy_is_distinct_and_preserves_file(self):
        before = self.path.read_bytes()
        with checkpoint_lock(self.path):
            child = self.cli()
        self.assertEqual(child.returncode, 3)
        report = json.loads(child.stdout)
        self.assertEqual(report["error"]["code"], "checkpoint_busy")
        self.assertFalse(report["checkpoint"]["execution_started"])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertNotIn("Traceback", child.stderr)

    def test_cli_corrupt_json_and_text_error(self):
        self.path.write_text("{}")
        for json_mode in (True, False):
            child = self.cli(json_mode)
            self.assertEqual(child.returncode, 3)
            self.assertNotIn("Traceback", child.stderr)
            if json_mode:
                self.assertEqual(
                    json.loads(child.stdout)["error"]["code"], "checkpoint_load_failed"
                )
            else:
                self.assertIn("checkpoint_load_failed", child.stdout)

    def test_validation_and_policy_errors_are_preexecution(self):
        before = self.path.read_bytes()
        report = run_checkpointed_report(self.runtime, "test", self.path, task_timeout_s=0)
        self.assertEqual(report["error"]["code"], "checkpoint_validate_failed")
        self.assertEqual(self.path.read_bytes(), before)
        save_budget(ActionBudget({}), self.path)
        report = self.run_report()
        self.assertEqual(report["error"]["code"], "checkpoint_policy_failed")
        self.assertFalse(report["checkpoint"]["execution_started"])
        self.runtime.run.assert_not_called()

    def test_unlock_failure_keeps_committed_state_but_no_success(self):
        from contextlib import contextmanager

        @contextmanager
        def failing_unlock(path):
            yield path
            raise OSError("unlock failed")

        with patch("embodied_agent.checkpointed_runtime.checkpoint_lock", failing_unlock):
            report = self.run_report()
        self.assertEqual(report["error"]["code"], "checkpoint_unlock_failed")
        self.assertEqual(report["checkpoint"]["checkpoint_state"], "committed")
        self.assertFalse(report["success"])
