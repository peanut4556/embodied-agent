import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from embodied_agent.wait_budget_store import WaitBudgetStore


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "budget.db"
        self.store = WaitBudgetStore.create(self.path, total_threshold_ns=10, slots=3)

    def test_same_pool_old_copy_rejected(self):
        old = Path(self.temp.name) / "old.db"
        shutil.copyfile(self.path, old)
        self.store.reserve("first")
        checkpoint = self.store.checkpoint()
        shutil.copyfile(old, self.path)
        with self.assertRaisesRegex(ValueError, "history differs"):
            WaitBudgetStore(self.path, expected_checkpoint=checkpoint)

    def test_prefix_allows_append_and_checkpoint_is_copied(self):
        self.store.reserve("first")
        checkpoint = self.store.checkpoint()
        pinned = WaitBudgetStore(self.path, expected_checkpoint=checkpoint)
        checkpoint["sha256"] = "0" * 64
        self.assertEqual(pinned.reserve("second")["slot"], 2)
        self.assertEqual(pinned.checkpoint()["reserved_slots"], 2)

    def test_same_count_history_rewrite_blocks_existing_object_reserve(self):
        self.store.reserve("first")
        pinned = WaitBudgetStore(self.path, expected_checkpoint=self.store.checkpoint())
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE reservations SET request='changed'")
        with self.assertRaisesRegex(ValueError, "history differs"):
            pinned.reserve("second")
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM reservations").fetchone()[0], 1)

    def test_configuration_change_and_invalid_checkpoint_rejected(self):
        checkpoint = self.store.checkpoint()
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE config SET total=11")
        with self.assertRaisesRegex(ValueError, "history differs"):
            WaitBudgetStore(self.path, expected_checkpoint=checkpoint)
        for bad in ({}, checkpoint | {"reserved_slots": True}, checkpoint | {"sha256": "x" * 64}):
            with self.assertRaisesRegex(ValueError, "invalid budget checkpoint"):
                WaitBudgetStore(self.path, expected_checkpoint=bad)


if __name__ == "__main__":
    unittest.main()
