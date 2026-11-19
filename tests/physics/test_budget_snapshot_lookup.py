import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agent.wait_budget_store import WaitBudgetStore


class SnapshotLookupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = WaitBudgetStore.create(self.root / "pool.db", total_threshold_ns=10, slots=3)

    def test_order_receipts_and_checkpoint_match_single_snapshot(self):
        receipt = self.store.reserve("a")
        result = self.store.lookup_many(["missing", "a"])
        self.assertTrue(result["single_database_snapshot"])
        self.assertEqual(result["results"][0]["result"]["status"], "not_found")
        self.assertEqual(result["results"][1]["result"]["receipt"], receipt)
        self.assertEqual(result["checkpoint"], self.store.checkpoint())

    def test_commit_during_read_does_not_mix_old_and_new_results(self):
        with sqlite3.connect(self.store.path) as db:
            db.execute("PRAGMA journal_mode=WAL")
        other = WaitBudgetStore(self.store.path)
        original = self.store._rows

        def concurrent_commit(db):
            rows = original(db)
            other.reserve("new")
            return rows

        with patch.object(self.store, "_rows", side_effect=concurrent_commit):
            batch = self.store.lookup_many(["new", "missing"])
        self.assertEqual(batch["checkpoint"]["reserved_slots"], 0)
        self.assertTrue(all(r["result"]["status"] == "not_found" for r in batch["results"]))
        self.assertEqual(self.store.lookup("new")["status"], "reserved")

    def test_invalid_batch_rejected_before_connection(self):
        for ids in ([], ["a"] * 2, ["x"] * 65, [True], "a"):
            with patch.object(self.store, "_connect") as connection, self.assertRaises(ValueError):
                self.store.lookup_many(ids)
            connection.assert_not_called()

    def test_pinned_history_failure_returns_no_partial_batch(self):
        self.store.reserve("a")
        pinned = WaitBudgetStore(self.store.path, expected_checkpoint=self.store.checkpoint())
        with sqlite3.connect(self.store.path) as db:
            db.execute("UPDATE reservations SET request='changed'")
        with self.assertRaisesRegex(ValueError, "history differs"):
            pinned.lookup_many(["a", "changed"])

    def test_cli_distinguishes_missing_and_found(self):
        self.store.reserve("a")
        requests = self.root / "requests.json"
        for ids, expected in ((["a"], 0), (["a", "missing"], 1)):
            requests.write_text(json.dumps(ids))
            child = subprocess.run(
                [
                    sys.executable,
                    "scripts/lookup_budget_batch.py",
                    "--database",
                    str(self.store.path),
                    "--expected-pool-id",
                    self.store.pool_id,
                    "--requests",
                    str(requests),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(child.returncode, expected)
            self.assertTrue(json.loads(child.stdout)["single_database_snapshot"])


if __name__ == "__main__":
    unittest.main()
