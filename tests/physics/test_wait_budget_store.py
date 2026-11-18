import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.wait_budget_store import WaitBudgetStore


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "budget.db"

    def test_reopen_preserves_integer_shares_and_rejects_replay(self):
        store = WaitBudgetStore.create(self.path, total_threshold_ns=10, slots=3)
        self.assertEqual(store.reserve("a")["allowance_ns"], 4)
        other = WaitBudgetStore(self.path)
        with self.assertRaisesRegex(RuntimeError, "replay"):
            other.reserve("a")
        self.assertEqual(other.reserve("b")["allowance_ns"], 3)
        self.assertEqual(other.reserve("c")["allowance_ns"], 3)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            store.reserve("d")
        self.assertEqual(store.snapshot()["remaining_ns"], 0)

    def test_missing_existing_and_invalid_config(self):
        with self.assertRaises(sqlite3.OperationalError):
            WaitBudgetStore(self.path)
        for total, slots in ((True, 1), (-1, 1), (1, 0), (1, True)):
            with self.assertRaises(ValueError):
                WaitBudgetStore.create(self.path, total_threshold_ns=total, slots=slots)
        WaitBudgetStore.create(self.path, total_threshold_ns=0, slots=1)
        with self.assertRaises(FileExistsError):
            WaitBudgetStore.create(self.path, total_threshold_ns=1, slots=1)

    def test_lock_failure_does_not_allocate(self):
        store = WaitBudgetStore.create(self.path, total_threshold_ns=10, slots=1)
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            with self.assertRaises(sqlite3.OperationalError):
                store.reserve("a")
        self.assertEqual(store.snapshot()["reserved_slots"], 0)

    def test_manager_reconstruction_cannot_replenish_and_no_spawn_on_refusal(self):
        store = WaitBudgetStore.create(self.path, total_threshold_ns=10_000_000, slots=1)

        def manager():
            return CaptureRecovery(
                position_error_m=(0.001,) * 3,
                capture_profile="reserved",
                persistent_wait_budget=WaitBudgetStore(self.path),
            )

        first = manager()
        with (
            patch(
                "embodied_agent.capture_recovery.ProcessPairCapture",
                side_effect=RuntimeError("spawn"),
            ) as spawn,
            self.assertRaisesRegex(RuntimeError, "spawn"),
        ):
            first.start(lambda: None)
        self.assertEqual(spawn.call_args.args[0].wait_cpu_budget_s, 0.01)
        self.assertEqual(store.snapshot()["reserved_slots"], 1)
        second = manager()
        with (
            patch("embodied_agent.capture_recovery.ProcessPairCapture") as spawn,
            self.assertRaisesRegex(RuntimeError, "exhausted"),
        ):
            second.start(lambda: None)
        spawn.assert_not_called()
        self.assertTrue(second.close())
        with self.assertRaisesRegex(RuntimeError, "new manager"):
            second.start(lambda: None)

    def test_real_failed_start_then_new_manager_uses_remaining_slot(self):
        from test_capture_recovery import bad_factory, factory

        store = WaitBudgetStore.create(self.path, total_threshold_ns=10_000_000, slots=2)
        first = CaptureRecovery(
            position_error_m=(0.001,) * 3, capture_profile="reserved", persistent_wait_budget=store
        )
        try:
            self.assertIsNone(first.start(bad_factory))
        finally:
            self.assertTrue(first.close())
        second = CaptureRecovery(
            position_error_m=(0.001,) * 3,
            capture_profile="reserved",
            persistent_wait_budget=WaitBudgetStore(self.path),
        )
        try:
            self.assertIsNotNone(second.start(factory))
            reservation = second.result()["wait_cpu_budget"]["last_persistent_reservation"]
            self.assertEqual(reservation["slot"], 2)
            self.assertEqual(reservation["allowance_ns"], 5_000_000)
        finally:
            self.assertTrue(second.close())
        self.assertEqual(store.snapshot()["reserved_slots"], 2)

    def test_identity_change_and_invalid_combinations(self):
        store = WaitBudgetStore.create(self.path, total_threshold_ns=10, slots=1)
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE config SET pool='changed'")
        with self.assertRaisesRegex(ValueError, "identity"):
            store.reserve("a")
        with self.assertRaises(ValueError):
            CaptureRecovery(position_error_m=(0.001,) * 3, persistent_wait_budget=store)
        with self.assertRaises(ValueError):
            CaptureRecovery(
                position_error_m=(0.001,) * 3,
                capture_profile="reserved",
                persistent_wait_budget=store,
                wait_cpu_budget_s=0.01,
            )


if __name__ == "__main__":
    unittest.main()
