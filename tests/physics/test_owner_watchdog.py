import time
import unittest

from test_process_pair_capture import factory, startup_hang_factory, stuck_factory

from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.process_pair_capture import ProcessPairCapture
from embodied_agent.two_stage_capture import StageCaptureRequest


class WatchdogTests(unittest.TestCase):
    def test_disconnect_exits_even_during_blocked_read(self):
        backend = ProcessPairCapture(stuck_factory, capacity=1, owner_watchdog=True)
        try:
            self.assertTrue(backend.wait_ready())
            backend.submit(StageCaptureRequest(1, "candidate", "test", 0, time.monotonic()))
            slot = backend.slots[0]
            slot["lifeline"].close()
            slot["process"].join(2)
            self.assertEqual(slot["process"].exitcode, 86)
        finally:
            backend.cancel()
            backend.reap(0.25)

    def test_disconnect_during_factory(self):
        backend = ProcessPairCapture(startup_hang_factory, capacity=1, owner_watchdog=True)
        try:
            slot = backend.slots[0]
            slot["lifeline"].close()
            slot["process"].join(2)
            self.assertEqual(slot["process"].exitcode, 86)
        finally:
            backend.cancel()
            backend.reap(0.25)

    def test_recovery_exposes_opt_in_and_normal_cleanup(self):
        manager = CaptureRecovery(position_error_m=(0.001,) * 3, owner_watchdog=True)
        try:
            self.assertIsNotNone(manager.start(factory))
            self.assertTrue(manager.backend.result()["owner_watchdog_enabled"])
        finally:
            self.assertTrue(manager.close())

    def test_nonboolean_rejected(self):
        with self.assertRaises(ValueError):
            ProcessPairCapture(factory, owner_watchdog=1)
        with self.assertRaises(ValueError):
            CaptureRecovery(position_error_m=(0.001,) * 3, owner_watchdog=1)


if __name__ == "__main__":
    unittest.main()
