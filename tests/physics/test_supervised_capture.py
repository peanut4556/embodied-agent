import os
import signal
import time
import unittest

from test_process_pair_capture import factory, stuck_factory

from embodied_agent.process_pair_capture import ProcessPairCapture
from embodied_agent.two_stage_capture import StageCaptureRequest


class SupervisedCaptureTests(unittest.TestCase):
    def test_frame_roundtrip_and_both_levels_reaped(self):
        backend = ProcessPairCapture(factory, capacity=1, supervised=True)
        try:
            self.assertTrue(backend.wait_ready())
            self.assertTrue(
                backend.submit(StageCaptureRequest(1, "candidate", "session", 0, time.monotonic()))
            )
            deadline = time.monotonic() + 2
            packets = []
            while not packets and time.monotonic() < deadline:
                packets = backend.drain()
                time.sleep(0.001)
            self.assertEqual(packets[0][1].session_id, "session")
            self.assertIsNone(packets[0][2])
        finally:
            backend.cancel()
            result = backend.reap(0.25)
        self.assertTrue(result["workers_stopped"])
        self.assertEqual(result["supervision"][0]["events"][-1]["status"], "stopped")

    def test_stopped_reader_reaped_without_killing_supervisor_first(self):
        backend = ProcessPairCapture(stuck_factory, capacity=1, supervised=True)
        try:
            self.assertTrue(backend.wait_ready())
            pid = backend.result()["supervision"][0]["events"][0]["worker_pid"]
            os.kill(pid, signal.SIGSTOP)
            time.sleep(0.02)
        finally:
            backend.cancel()
            result = backend.reap(0.25)
        self.assertTrue(result["workers_stopped"])
        self.assertEqual(result["supervision"][0]["events"][-1]["worker_exitcode"], -signal.SIGKILL)
        self.assertFalse(result["workers"][0]["killed"])

    def test_invalid_option(self):
        with self.assertRaises(ValueError):
            ProcessPairCapture(factory, supervised=1)


if __name__ == "__main__":
    unittest.main()
