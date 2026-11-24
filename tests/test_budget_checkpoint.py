import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import LIMIT, canonical, load_budget, save_budget
from embodied_agent.runtime import AgentRuntime
from scripts.evaluate_budget_drift import Fixture


class BudgetCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state.json"
        budget = ActionBudget(
            {"verify": [0.01] * 20, "pick": [0.01] * 20}, drift_policy="quarantine"
        )
        budget.record("verify", 0.04, confirmed=True)
        budget.record("pick", 0.04, confirmed=True)
        self.budget = budget.validated_recalibrated(
            "verify",
            [0.03] * 20,
            batch_id="train",
            validation_samples=[0.032] * 20,
            validation_batch_id="holdout",
        )
        save_budget(self.budget, self.path)

    def tamper(self, mutate, *, rehash=True):
        data = json.loads(self.path.read_text())
        mutate(data["state"])
        if rehash:
            data["sha256"] = hashlib.sha256(canonical(data["state"]).encode()).hexdigest()
        self.path.write_text(json.dumps(data))

    def test_roundtrip_and_runtime_blocking(self):
        restored = load_budget(self.path)
        self.assertEqual(restored.recalibration_history(), self.budget.recalibration_history())
        self.assertEqual(restored.revision, 1)
        self.assertTrue(restored.blocked("pick"))
        self.assertFalse(restored.blocked("verify"))
        fixture = Fixture("pick", 0.01, 0.08)
        result = AgentRuntime(fixture, fixture, fixture, clock=lambda: fixture.t).run(
            "test", task_timeout_s=0.1, action_budget=restored
        )
        self.assertFalse(result.success)
        self.assertEqual(fixture.calls, 0)
        with self.assertRaises(ValueError):
            restored.recalibrated("pick", [0.03] * 20, batch_id="holdout")
        recovered = restored.recalibrated("pick", [0.03] * 20, batch_id="new")
        save_budget(recovered, self.path)
        self.assertFalse(load_budget(self.path).blocked("pick"))

    def test_real_new_process_preserves_quarantine(self):
        code = (
            "import sys; from embodied_agent.budget_checkpoint import load_budget; "
            'b=load_budget(sys.argv[1]); assert b.blocked("pick"); '
            'assert b.revision==1; assert not b.blocked("verify")'
        )
        child = subprocess.run(
            [sys.executable, "-c", code, str(self.path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_corruption_and_missing_file_do_not_create_default(self):
        self.tamper(lambda state: state.update(degraded=[]), rehash=False)
        with self.assertRaises(ValueError):
            load_budget(self.path)
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_budget(self.path)

    def test_structural_invariants_even_with_recomputed_checksum(self):
        mutations = (
            lambda s: s.update(schema=True),
            lambda s: s.update(revision=2),
            lambda s: s.update(degraded=["pick", "pick"]),
            lambda s: s["estimates"].update(verify=0.2),
            lambda s: s["history"][0].update(validation_max_s=0.3),
            lambda s: s["history"][0].update(validation_batch_id="train"),
            lambda s: s.update(extra=1),
        )
        for mutate in mutations:
            save_budget(self.budget, self.path)
            self.tamper(mutate)
            with self.assertRaises(ValueError):
                load_budget(self.path)

    def test_parser_rejects_ambiguous_nonfinite_large_and_nested_data(self):
        for raw in (
            '{"state":{},"state":{}}',
            '{"x":NaN}',
            "[" * 100 + "0" + "]" * 100,
            " " * (LIMIT + 1),
        ):
            self.path.write_text(raw)
            with self.assertRaises(ValueError):
                load_budget(self.path)

    def test_failed_replace_preserves_old_file_and_cleans_temporary(self):
        before = self.path.read_bytes()
        with (
            patch("embodied_agent.budget_checkpoint.os.replace", side_effect=OSError("injected")),
            self.assertRaises(OSError),
        ):
            save_budget(self.budget, self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob(".budget-*")), [])

    def test_invalid_memory_state_cannot_overwrite_checkpoint(self):
        before = self.path.read_bytes()
        self.budget.estimates["pick"] = float("inf")
        with self.assertRaises(ValueError):
            save_budget(self.budget, self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_read_only_inspection_cli(self):
        before = self.path.read_bytes()
        child = subprocess.run(
            [sys.executable, "-m", "embodied_agent.budget_checkpoint", str(self.path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout)["degraded"], ["pick"])
        self.assertEqual(self.path.read_bytes(), before)
        self.path.write_text("{}")
        child = subprocess.run(
            [sys.executable, "-m", "embodied_agent.budget_checkpoint", str(self.path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(child.returncode, 2)
        self.assertEqual(child.stdout, "")
        self.assertFalse(json.loads(child.stderr)["valid"])
