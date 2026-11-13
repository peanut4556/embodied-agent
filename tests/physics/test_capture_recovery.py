import time
import unittest
from unittest.mock import patch

from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.threaded_pair_capture import CapturedPair


class Reader:
    def __call__(self, req):
        p = StampedPosition((0.64, 0, 0.045), time.monotonic())
        return CapturedPair(req.session_id, p, p)


def factory():
    return Reader()


def bad_factory():
    raise RuntimeError("camera failed")


class RecoveryTests(unittest.TestCase):
    def test_active_task_cannot_be_replayed(self):
        m = CaptureRecovery(position_error_m=(0.001,) * 3)
        try:
            token = m.start(factory)
            self.assertIsNotNone(token)
            with self.assertRaises(RuntimeError):
                m.start(factory)
            self.assertEqual(m.attempts, 1)
        finally:
            self.assertTrue(m.close())

    def test_retire_then_new_session_without_old_evidence(self):
        m = CaptureRecovery(position_error_m=(0.001,) * 3)
        try:
            old = m.start(factory)
            old_backend = m.backend
            self.assertEqual(m.step(old, stop=True), "cancelled")
            new = m.start(factory)
            self.assertNotEqual(old, new)
            self.assertTrue(old_backend.result()["workers_stopped"])
            self.assertTrue(old_backend.result()["closed"])
            self.assertEqual(m.runtime.controller.requests, [])
            self.assertIsNone(m.runtime.controller.gate)
            self.assertIsNone(m.step(old, stop=True))
            self.assertEqual(m.runtime.controller.status, "pending")
            self.assertEqual(len(m.history), 1)
            self.assertEqual(old_backend.drain(), [])
        finally:
            self.assertTrue(m.close())

    def test_cleanup_pending_never_spawns_replacement(self):
        m = CaptureRecovery(position_error_m=(0.001,) * 3)
        try:
            token = m.start(factory)
            m.step(token, stop=True)
            with (
                patch.object(
                    m.runtime, "close", return_value={"closed": True, "workers_stopped": False}
                ),
                patch("embodied_agent.capture_recovery.ProcessPairCapture") as spawn,
            ):
                self.assertIsNone(m.start(factory))
                spawn.assert_not_called()
            self.assertEqual(m.attempts, 1)
            self.assertEqual(m.state, "cleanup_pending")
        finally:
            self.assertTrue(m.close())

    def test_failed_start_counts_against_budget(self):
        m = CaptureRecovery(position_error_m=(0.001,) * 3, max_generations=2)
        try:
            self.assertIsNone(m.start(bad_factory))
            self.assertEqual(len(m.history), 1)
            self.assertIsNotNone(m.start(factory))
            m.close()
            with self.assertRaises(RuntimeError):
                m.start(factory)
            self.assertEqual(len(m.history), 2)
        finally:
            m.close()

    def test_unknown_spawn_ownership_latches(self):
        m = CaptureRecovery(position_error_m=(0.001,) * 3)
        with (
            patch(
                "embodied_agent.capture_recovery.ProcessPairCapture",
                side_effect=RuntimeError("spawn failed"),
            ),
            self.assertRaises(RuntimeError),
        ):
            m.start(factory)
        self.assertEqual(m.state, "ownership_uncertain")
        with self.assertRaises(RuntimeError):
            m.start(factory)
        self.assertFalse(m.close())

    def test_snapshot_and_limit_validation(self):
        m = CaptureRecovery(position_error_m=(0.001,) * 3)
        r = m.result()
        r["history"].append({})
        self.assertEqual(m.history, [])
        for n in (0, 4, True):
            with self.assertRaises(ValueError):
                CaptureRecovery(position_error_m=(0.001,) * 3, max_generations=n)

    def test_runtime_exception_preserves_terminal_lifecycle(self):
        m = CaptureRecovery(position_error_m=(0.001,) * 3)
        try:
            token = m.start(factory)
            m.runtime.clock = lambda: -1
            with self.assertRaises(ValueError):
                m.step(token)
            self.assertEqual(m.state, "terminal")
            self.assertEqual(m.runtime.controller.reason, "capture_runtime_exception")
        finally:
            self.assertTrue(m.close())
