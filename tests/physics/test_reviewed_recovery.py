import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_capture_recovery import factory

from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.wait_budget_store import WaitBudgetStore


class ReviewedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WaitBudgetStore.create(
            Path(self.temp.name) / "pool.db", total_threshold_ns=10_000_000, slots=3
        )
        self.manager = CaptureRecovery(
            position_error_m=(0.001,) * 3,
            capture_profile="reserved",
            persistent_wait_budget=self.store,
            require_reviewed_budget=True,
        )
        self.addCleanup(self.manager.close)

    def test_missing_review_cannot_spawn_or_spend_attempt(self):
        with (
            patch("embodied_agent.capture_recovery.ProcessPairCapture") as spawn,
            self.assertRaisesRegex(ValueError, "every start"),
        ):
            self.manager.start(factory)
        spawn.assert_not_called()
        self.assertEqual(self.manager.attempts, 0)
        self.assertEqual(self.store.snapshot()["reserved_slots"], 0)

    def test_stale_review_blocks_backend_and_latches(self):
        review = self.store.checkpoint()
        self.store.reserve("external")
        with (
            patch("embodied_agent.capture_recovery.ProcessPairCapture") as spawn,
            self.assertRaisesRegex(RuntimeError, "stale"),
        ):
            self.manager.start(factory, reviewed_budget_checkpoint=review)
        spawn.assert_not_called()
        self.assertEqual(self.store.snapshot()["reserved_slots"], 1)
        self.manager.close()
        with self.assertRaisesRegex(RuntimeError, "new manager"):
            self.manager.start(factory, reviewed_budget_checkpoint=self.store.checkpoint())

    def test_real_restart_requires_fresh_explicit_review(self):
        token = self.manager.start(factory, reviewed_budget_checkpoint=self.store.checkpoint())
        self.assertIsNotNone(token)
        self.manager.step(token, stop=True)
        with self.assertRaisesRegex(ValueError, "every start"):
            self.manager.start(factory)
        self.assertEqual(self.manager.attempts, 1)
        fresh = self.store.checkpoint()
        new = self.manager.start(factory, reviewed_budget_checkpoint=fresh)
        self.assertIsNotNone(new)
        self.assertNotEqual(new, token)
        self.assertEqual(self.store.snapshot()["reserved_slots"], 2)
        self.assertTrue(self.manager.result()["require_reviewed_budget"])

    def test_invalid_mode_or_unbacked_review_rejected(self):
        for value in (True, 1):
            with self.assertRaises(ValueError):
                CaptureRecovery(position_error_m=(0.001,) * 3, require_reviewed_budget=value)
        plain = CaptureRecovery(position_error_m=(0.001,) * 3)
        with self.assertRaisesRegex(ValueError, "persistent pool"):
            plain.start(factory, reviewed_budget_checkpoint=self.store.checkpoint())
        self.assertEqual(plain.attempts, 0)


if __name__ == "__main__":
    unittest.main()
