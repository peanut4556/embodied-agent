import contextlib
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from embodied_agent import cli
from embodied_agent.models import TaskResult


class TaskBudgetCliTests(unittest.TestCase):
    def test_budget_forwarded_and_failure_exit_for_both_formats(self):
        for json_mode in (False, True):
            args = ["agent", "red block box", "--task-timeout", "0.25"]
            if json_mode:
                args.append("--json")
            with patch.object(sys, "argv", args), patch.object(cli, "AgentRuntime") as runtime:
                runtime.return_value.run.return_value = TaskResult(
                    False, "task deadline exceeded", 0
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(cli.main(), 1)
                runtime.return_value.run.assert_called_once_with(
                    "red block box", task_timeout_s=0.25
                )
                if json_mode:
                    self.assertFalse(json.loads(output.getvalue())["success"])

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "embodied_agent", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_real_module_success_and_deadline_exit(self):
        success = self.run_cli("red block box", "--json", "--task-timeout", "5")
        self.assertEqual(success.returncode, 0)
        self.assertTrue(json.loads(success.stdout)["success"])
        # Below the resolution at normal monotonic uptime: planning must exhaust it.
        failure = self.run_cli("red block box", "--json", "--task-timeout", "1e-12")
        self.assertEqual(failure.returncode, 1)
        result = json.loads(failure.stdout)
        self.assertFalse(result["success"])
        self.assertEqual(result["completed_steps"], 0)
        self.assertEqual(result["confirmation_reports"], [])

    def test_invalid_budget_is_usage_error(self):
        for value in ("0", "-1", "nan", "inf", "abc"):
            child = self.run_cli("--task-timeout", value)
            self.assertEqual(child.returncode, 2)
            self.assertEqual(child.stdout, "")
            self.assertNotIn("Traceback", child.stderr)
