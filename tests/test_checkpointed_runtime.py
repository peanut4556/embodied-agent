import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import load_budget, save_budget
from embodied_agent.checkpointed_runtime import checkpoint_lock, run_checkpointed
from embodied_agent.runtime import AgentRuntime
from scripts.evaluate_budget_drift import Fixture


class CheckpointedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "budget.json"
        save_budget(ActionBudget({"verify": [0.01] * 20}, drift_policy="quarantine"), self.path)

    def runtime(self, duration=0.01):
        fixture = Fixture("verify", duration, 0.08)
        return AgentRuntime(fixture, fixture, fixture, clock=lambda: fixture.t)

    def child(self, code):
        return subprocess.run(
            [sys.executable, "-c", code, str(self.path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_success_commits_and_drift_persists(self):
        self.assertTrue(
            run_checkpointed(self.runtime(), "test", self.path, task_timeout_s=0.1).success
        )
        self.assertFalse(load_budget(self.path).blocked("pick"))
        run_checkpointed(self.runtime(0.03), "test", self.path, task_timeout_s=0.1)
        self.assertTrue(load_budget(self.path).blocked("verify"))
        runtime = self.runtime()
        result = run_checkpointed(runtime, "test", self.path, task_timeout_s=0.1)
        self.assertFalse(result.success)
        self.assertEqual(runtime.robot.calls, 0)

    def test_abrupt_process_death_leaves_prepared_quarantine(self):
        child = self.child("""import os, sys
from embodied_agent.checkpointed_runtime import run_checkpointed
class Runtime:
    def run(self, *args, **kwargs): os._exit(23)
run_checkpointed(Runtime(), 'test', sys.argv[1], task_timeout_s=1)
""")
        self.assertEqual(child.returncode, 23, child.stderr)
        restored = load_budget(self.path)
        self.assertTrue(all(restored.blocked(a) for a in ("locate", "pick", "place", "verify")))
        # OS releases the lock after death; next caller can load but cannot dispatch.
        runtime = self.runtime()
        self.assertFalse(run_checkpointed(runtime, "test", self.path, task_timeout_s=0.1).success)
        self.assertEqual(runtime.robot.calls, 0)

    def test_preparation_failure_prevents_run(self):
        runtime = Mock()
        before = self.path.read_bytes()
        with (
            patch("embodied_agent.checkpointed_runtime.save_budget", side_effect=OSError("disk")),
            self.assertRaises(OSError),
        ):
            run_checkpointed(runtime, "test", self.path, task_timeout_s=1)
        runtime.run.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_final_save_failure_never_returns_success(self):
        count = 0

        def save(budget, path):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("disk")
            save_budget(budget, path)

        with (
            patch("embodied_agent.checkpointed_runtime.save_budget", side_effect=save),
            self.assertRaises(OSError),
        ):
            run_checkpointed(self.runtime(), "test", self.path, task_timeout_s=1)
        self.assertTrue(load_budget(self.path).blocked("verify"))

    def test_runtime_exception_preserves_uncertainty(self):
        runtime = Mock()
        runtime.run.side_effect = RuntimeError("device failed")
        with self.assertRaises(RuntimeError):
            run_checkpointed(runtime, "test", self.path, task_timeout_s=1)
        self.assertTrue(load_budget(self.path).blocked("pick"))

    def test_cross_process_contention_refuses_without_write(self):
        before = self.path.read_bytes()
        with checkpoint_lock(self.path):
            child = self.child("""import sys
from embodied_agent.checkpointed_runtime import checkpoint_lock, CheckpointBusy
try:
    with checkpoint_lock(sys.argv[1]): raise AssertionError('acquired')
except CheckpointBusy: pass
""")
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_config_and_missing_checkpoint_no_run(self):
        runtime = Mock()
        with self.assertRaises(ValueError):
            run_checkpointed(runtime, "test", self.path, task_timeout_s=0)
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            run_checkpointed(runtime, "test", self.path, task_timeout_s=1)
        runtime.run.assert_not_called()

    def test_cli_checkpoint_path(self):
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "embodied_agent",
                "red block box",
                "--budget-checkpoint",
                str(self.path),
                "--task-timeout",
                "5",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertTrue(json.loads(child.stdout)["success"])
        self.assertEqual(load_budget(self.path).drift_policy, "quarantine")

    def test_fallback_policy_is_not_silently_changed(self):
        save_budget(ActionBudget({}), self.path)
        before = self.path.read_bytes()
        runtime = Mock()
        with self.assertRaises(ValueError):
            run_checkpointed(runtime, "test", self.path, task_timeout_s=1)
        runtime.run.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_cli_requires_explicit_timeout(self):
        before = self.path.read_bytes()
        child = subprocess.run(
            [sys.executable, "-m", "embodied_agent", "--budget-checkpoint", str(self.path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(child.returncode, 2)
        self.assertEqual(self.path.read_bytes(), before)
