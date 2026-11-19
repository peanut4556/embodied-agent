import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embodied_agent.wait_budget_store import WaitBudgetStore


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "pool.db"
        self.store = WaitBudgetStore.create(self.path, total_threshold_ns=10, slots=3)

    def cli(self, identity, pool_id=None, checkpoint=None):
        command = [
            sys.executable,
            "scripts/lookup_wait_budget.py",
            "--database",
            str(self.path),
            "--expected-pool-id",
            pool_id or self.store.pool_id,
            "--request-id",
            identity,
        ]
        if checkpoint:
            command += ["--checkpoint", str(checkpoint)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return result.returncode, json.loads(result.stdout)

    def test_historical_receipt_unchanged_after_later_allocations(self):
        first = self.store.reserve("a")
        self.store.reserve("b")
        before = self.store.snapshot()
        result = self.store.lookup("a")
        self.assertEqual(result["receipt"], first)
        self.assertFalse(result["execution_verified"])
        self.assertEqual(self.store.snapshot(), before)
        with self.assertRaisesRegex(RuntimeError, "replay"):
            self.store.reserve("a")

    def test_missing_is_not_a_new_reservation(self):
        self.assertEqual(self.cli("missing")[0], 1)
        self.assertEqual(self.store.snapshot()["reserved_slots"], 0)

    def test_committed_then_process_exit_reconciles_without_replay(self):
        code = (
            "import os,sys; from embodied_agent.wait_budget_store import WaitBudgetStore; "
            'WaitBudgetStore(sys.argv[1]).reserve("lost"); os._exit(73)'
        )
        child = subprocess.run(
            [sys.executable, "-c", code, str(self.path)], timeout=10, check=False
        )
        self.assertEqual(child.returncode, 73)
        status, result = self.cli("lost")
        self.assertEqual(status, 0)
        self.assertEqual(result["receipt"]["slot"], 1)
        self.assertEqual(self.store.snapshot()["reserved_slots"], 1)

    def test_wrong_pool_and_bad_checkpoint_are_unconfirmed(self):
        self.assertEqual(self.cli("a", pool_id="wrong")[0], 2)
        checkpoint = Path(self.temp.name) / "bad.json"
        checkpoint.write_text("{")
        code, result = self.cli("a", checkpoint=checkpoint)
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "unconfirmed")

    def test_invalid_identity_rejected(self):
        for identity in ("", True, "x" * 257):
            with self.assertRaises(ValueError):
                self.store.lookup(identity)


if __name__ == "__main__":
    unittest.main()
