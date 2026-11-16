import unittest

from embodied_agent.capture_recovery import CaptureRecovery
from scripts.check_process_pair_capture import fast_factory
from scripts.check_supervised_recovery import blocked_factory


class SupervisedRecoveryTests(unittest.TestCase):
    def test_deadline_cleanup_precedes_explicit_new_session(self):
        manager = CaptureRecovery(
            position_error_m=(0.001,) * 3, max_generations=2, supervised=True, runtime_limit_s=0.2
        )
        try:
            self.assertIsNone(manager.start(blocked_factory))
            old = manager.session
            history = manager.result()["history"]
            self.assertEqual(len(history), 1)
            self.assertTrue(history[0]["cleanup"]["workers_stopped"])
            self.assertTrue(
                any(
                    d["events"][-1].get("runtime_limit_exceeded")
                    for d in history[0]["cleanup"]["supervision"]
                )
            )
            new = manager.start(fast_factory)
            self.assertIsNotNone(new)
            self.assertNotEqual(old, new)
            self.assertEqual(manager.consume_confirmation(old)["reason"], "session_mismatch")
            self.assertIsNone(manager.step(old))
        finally:
            self.assertTrue(manager.close())
        with self.assertRaisesRegex(RuntimeError, "generation budget exhausted"):
            manager.start(fast_factory)


if __name__ == "__main__":
    unittest.main()
