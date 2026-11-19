import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embodied_agent.budget_start_intent import save_intent
from embodied_agent.wait_budget_store import WaitBudgetStore


class IntentCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = WaitBudgetStore.create(self.root / "pool.db", total_threshold_ns=10, slots=2)
        self.intent = self.root / "intent.json"

    def run_cli(self, *extra):
        return subprocess.run(
            [sys.executable, "scripts/lookup_wait_budget.py", "--intent", str(self.intent), *extra],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_child_exit_recovered_from_intent_path_only(self):
        code = """import os,sys
from embodied_agent.wait_budget_store import WaitBudgetStore
from embodied_agent.budget_start_intent import save_intent
store = WaitBudgetStore(sys.argv[1])
save_intent(sys.argv[2], store, 'lost')
store.reserve('lost')
os._exit(73)
"""
        child = subprocess.run(
            [sys.executable, "-c", code, str(self.store.path), str(self.intent)],
            timeout=10,
            check=False,
        )
        self.assertEqual(child.returncode, 73)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["receipt"]["request_id"], "lost")
        self.assertEqual(self.store.snapshot()["reserved_slots"], 1)

    def test_unsubmitted_intent_not_found_without_allocation(self):
        save_intent(self.intent, self.store, "pending")
        result = self.run_cli()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["status"], "not_found")
        self.assertEqual(self.store.snapshot()["reserved_slots"], 0)

    def test_independent_checkpoint_turns_rollback_into_unconfirmed(self):
        old = self.root / "old.db"
        shutil.copyfile(self.store.path, old)
        save_intent(self.intent, self.store, "lost")
        self.store.reserve("lost")
        baseline = self.root / "baseline.json"
        baseline.write_text(json.dumps(self.store.checkpoint()))
        shutil.copyfile(old, self.store.path)
        result = self.run_cli("--checkpoint", str(baseline))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "unconfirmed")

    def test_missing_or_corrupt_intent_is_unconfirmed(self):
        self.assertEqual(json.loads(self.run_cli().stdout)["status"], "unconfirmed")
        self.intent.write_text("{")
        self.assertEqual(self.run_cli().returncode, 2)

    def test_manual_override_is_rejected(self):
        save_intent(self.intent, self.store, "a")
        result = self.run_cli("--request-id", "different")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(self.store.snapshot()["reserved_slots"], 0)


if __name__ == "__main__":
    unittest.main()
