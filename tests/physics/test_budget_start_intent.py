import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agent.budget_start_intent import reconcile_intent
from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.wait_budget_store import WaitBudgetStore


class IntentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = WaitBudgetStore.create(self.root / "pool.db", total_threshold_ns=10, slots=2)
        self.intent = self.root / "intent.json"

    def test_process_exit_before_and_after_commit_reconciled_from_file_only(self):
        for after in (False, True):
            intent = self.root / f"intent-{after}.json"
            code = """import os,sys
from unittest.mock import patch
from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.wait_budget_store import WaitBudgetStore
store = WaitBudgetStore(sys.argv[1])
original = store.reserve
def interrupted(*a, **kw):
    if sys.argv[3] == 'yes': original(*a, **kw)
    os._exit(73)
m = CaptureRecovery(position_error_m=(.001,)*3, capture_profile='reserved', persistent_wait_budget=store)
with patch.object(store, 'reserve', interrupted):
    m.start(lambda: None, budget_intent_path=sys.argv[2])
"""
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    code,
                    str(self.store.path),
                    str(intent),
                    "yes" if after else "no",
                ],
                timeout=10,
                check=False,
            )
            self.assertEqual(child.returncode, 73)
            result = reconcile_intent(intent)
            self.assertEqual(result["status"], "reserved" if after else "not_found")
            self.assertFalse(result["execution_verified"])
        self.assertEqual(self.store.snapshot()["reserved_slots"], 1)

    def test_existing_intent_blocks_allocation_and_backend(self):
        self.intent.write_text("preserve")
        m = CaptureRecovery(
            position_error_m=(0.001,) * 3,
            capture_profile="reserved",
            persistent_wait_budget=self.store,
        )
        with (
            patch("embodied_agent.capture_recovery.ProcessPairCapture") as spawn,
            self.assertRaises(FileExistsError),
        ):
            m.start(lambda: None, budget_intent_path=self.intent)
        spawn.assert_not_called()
        self.assertEqual(self.store.snapshot()["reserved_slots"], 0)
        self.assertEqual(self.intent.read_text(), "preserve")

    def test_malformed_intent_refused(self):
        for value in ("{", json.dumps({"version": True}), "x" * 8193):
            self.intent.write_text(value)
            with self.assertRaises(ValueError):
                reconcile_intent(self.intent)

    def test_intent_requires_persistent_pool(self):
        m = CaptureRecovery(position_error_m=(0.001,) * 3)
        with self.assertRaises(ValueError):
            m.start(lambda: None, budget_intent_path=self.intent)
        self.assertFalse(self.intent.exists())


if __name__ == "__main__":
    unittest.main()
