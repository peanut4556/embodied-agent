import time
import unittest

from test_capture_supervisor import block
from test_process_pair_capture import factory

from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.capture_supervisor import SupervisedCaptureJob
from embodied_agent.process_pair_capture import ProcessPairCapture


class RuntimeLimitTests(unittest.TestCase):
    def test_owner_remains_alive_but_job_times_out(self):
        job = SupervisedCaptureJob(block, runtime_limit_s=0.2)
        try:
            job.wait_started()
            job._process.join(2)
            result = job.snapshot()
            self.assertFalse(result["supervisor_alive"])
            self.assertTrue(result["events"][-1]["runtime_limit_exceeded"])
            self.assertFalse(result["events"][-1]["owner_connection_closed"])
        finally:
            job.close()

    def test_backend_limit_and_no_new_work(self):
        backend = ProcessPairCapture(factory, capacity=1, supervised=True, runtime_limit_s=0.3)
        try:
            self.assertTrue(backend.wait_ready())
            time.sleep(0.5)
            result = backend.reap(0.25)
            self.assertTrue(result["workers_stopped"])
            self.assertTrue(result["supervision"][0]["events"][-1]["runtime_limit_exceeded"])
        finally:
            backend.cancel()
            backend.reap(0.25)

    def test_invalid_configuration(self):
        for value in (True, 0, -1, float("nan"), float("inf"), 61):
            with self.assertRaises(ValueError):
                SupervisedCaptureJob(block, runtime_limit_s=value)
        with self.assertRaises(ValueError):
            ProcessPairCapture(factory, runtime_limit_s=1)
        with self.assertRaises(ValueError):
            CaptureRecovery(position_error_m=(0.001,) * 3, runtime_limit_s=1)


if __name__ == "__main__":
    unittest.main()
