import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import save_budget
from embodied_agent.models import TaskResult
from embodied_agent.operation_receipt import _write, run_operation


class OperationReceiptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "budget.json"
        save_budget(ActionBudget({}, drift_policy="quarantine"), self.path)
        self.runtime = Mock()
        self.runtime.run.return_value = TaskResult(True, "done", 1)

    def invoke(self, **kwargs):
        options = {
            "operation_id": "request-1",
            "task_timeout_s": 1,
            "request_context": {"policy": "v1"},
        }
        options.update(kwargs)
        return run_operation(self.runtime, "test", self.path, **options)

    def test_completed_replay_does_not_call_runtime_again(self):
        first = self.invoke()
        before = self.path.read_bytes()
        second = self.invoke()
        self.assertTrue(first["success"])
        self.assertTrue(second["operation"]["replayed"])
        self.assertEqual(second["task_result"], first["task_result"])
        self.assertEqual(self.runtime.run.call_count, 1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_changed_context_or_timeout_conflicts(self):
        self.invoke()
        for options in ({"request_context": {"policy": "v2"}}, {"task_timeout_s": 2}):
            self.assertEqual(self.invoke(**options)["error"]["code"], "operation_conflict")
        self.assertEqual(self.runtime.run.call_count, 1)

    def test_task_failure_is_also_replayed(self):
        self.runtime.run.return_value = TaskResult(False, "not confirmed", 0)
        self.assertFalse(self.invoke()["success"])
        self.assertTrue(self.invoke()["operation"]["replayed"])
        self.assertEqual(self.runtime.run.call_count, 1)

    def test_runtime_failure_is_not_retried(self):
        self.runtime.run.side_effect = RuntimeError("private")
        self.assertIsNotNone(self.invoke()["error"])
        report = self.invoke()
        self.assertEqual(report["error"]["code"], "operation_in_doubt")
        self.assertTrue(report["error"]["action_outcome_unknown"])
        self.assertFalse(report["checkpoint"]["execution_started"])
        self.assertEqual(self.runtime.run.call_count, 1)

    def test_receipt_commit_failure_blocks_retry(self):
        calls = 0

        def write(path, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk")
            _write(path, data)

        with patch("embodied_agent.operation_receipt._write", side_effect=write):
            report = self.invoke()
        self.assertEqual(report["error"]["code"], "operation_commit_failed")
        self.assertEqual(self.invoke()["error"]["code"], "operation_in_doubt")
        self.assertEqual(self.runtime.run.call_count, 1)

    def test_registration_failure_prevents_execution(self):
        with patch("embodied_agent.operation_receipt._write", side_effect=OSError("disk")):
            self.assertIsNotNone(self.invoke()["error"])
        self.runtime.run.assert_not_called()

    def test_corrupt_receipt_does_not_fall_back_to_execution(self):
        self.invoke()
        receipt = next(self.path.with_name(self.path.name + ".operations").glob("*.json"))
        receipt.write_text("{}")
        report = self.invoke()
        self.assertIsNotNone(report["error"])
        self.assertTrue(report["error"]["action_outcome_unknown"])
        self.assertFalse(report["checkpoint"]["execution_started"])
        self.assertEqual(self.runtime.run.call_count, 1)

    def test_real_process_death_blocks_retry(self):
        code = """import os, sys
from embodied_agent.operation_receipt import run_operation
class Runtime:
    def run(self, *a, **kw): os._exit(23)
run_operation(Runtime(), 'test', sys.argv[1], operation_id='request-1', task_timeout_s=1, request_context={'policy':'v1'})
"""
        child = subprocess.run(
            [sys.executable, "-c", code, str(self.path)],
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(child.returncode, 23)
        self.assertEqual(self.invoke()["error"]["code"], "operation_in_doubt")
        self.runtime.run.assert_not_called()

    def test_cli_replays_across_processes(self):
        args = [
            sys.executable,
            "-m",
            "embodied_agent",
            "red block box",
            "--budget-checkpoint",
            str(self.path),
            "--task-timeout",
            "5",
            "--operation-id",
            "cli-1",
            "--json",
        ]
        for replayed in (False, True):
            child = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(json.loads(child.stdout)["operation"]["replayed"], replayed)

    def test_invalid_operation_id_prevents_execution(self):
        for value in ("", " x ", "x" * 129):
            self.assertIsNotNone(self.invoke(operation_id=value)["error"])
        self.runtime.run.assert_not_called()
