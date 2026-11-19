import sqlite3
import tempfile
import unittest
from pathlib import Path

from embodied_agent.wait_budget_store import WaitBudgetStore


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "pool.db"
        self.store = WaitBudgetStore.create(self.path, total_threshold_ns=10, slots=2)

    def test_pinned_reopen_preserves_spent_slot(self):
        self.store.reserve("first")
        reopened = WaitBudgetStore(self.path, expected_pool_id=self.store.pool_id)
        self.assertEqual(reopened.reserve("second")["slot"], 2)

    def test_different_pool_at_same_path_refused(self):
        replacement = Path(self.temp.name) / "replacement.db"
        WaitBudgetStore.create(replacement, total_threshold_ns=10, slots=2)
        replacement.replace(self.path)
        with self.assertRaisesRegex(ValueError, "unexpected budget pool"):
            WaitBudgetStore(self.path, expected_pool_id=self.store.pool_id)
        # Compatibility mode deliberately does not pin the original identity.
        self.assertNotEqual(WaitBudgetStore(self.path).pool_id, self.store.pool_id)

    def test_bad_identity_input(self):
        for value in ("", True, 3, "x" * 257):
            with self.assertRaises(ValueError):
                WaitBudgetStore(self.path, expected_pool_id=value)

    def test_invalid_existing_requests_refused_on_open_and_reserve(self):
        self.store.reserve("first")
        for value in (b"binary", "x" * 257, ""):
            with sqlite3.connect(self.path) as db:
                db.execute("UPDATE reservations SET request=?", (value,))
            with self.assertRaisesRegex(ValueError, "invalid reservations"):
                WaitBudgetStore(self.path, expected_pool_id=self.store.pool_id)
            with self.assertRaisesRegex(ValueError, "invalid reservations"):
                self.store.reserve("next")
            with sqlite3.connect(self.path) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM reservations").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
