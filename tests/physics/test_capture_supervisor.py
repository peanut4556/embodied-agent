import os
import signal
import time
import unittest

from embodied_agent.capture_supervisor import SupervisedCaptureJob


def block():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(0.01)


def finish():
    return


class SupervisorTests(unittest.TestCase):
    def test_stopped_worker_is_killed_and_reaped(self):
        job = SupervisedCaptureJob(block)
        try:
            started = job.wait_started()
            pid = started["events"][0]["worker_pid"]
            time.sleep(0.1)
            os.kill(pid, signal.SIGSTOP)
            time.sleep(0.02)
            result = job.close()
            self.assertFalse(result["supervisor_alive"])
            self.assertEqual(result["events"][-1]["worker_exitcode"], -signal.SIGKILL)
            self.assertEqual(result["events"][-1]["status"], "stopped")
        finally:
            job.close()

    def test_normal_completion(self):
        job = SupervisedCaptureJob(finish)
        job.wait_started()
        time.sleep(0.1)
        result = job.close()
        self.assertFalse(result["supervisor_alive"])
        self.assertEqual(result["events"][-1]["worker_exitcode"], 0)

    def test_invalid_job(self):
        with self.assertRaises(TypeError):
            SupervisedCaptureJob(None)


if __name__ == "__main__":
    unittest.main()
