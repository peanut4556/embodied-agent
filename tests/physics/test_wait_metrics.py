import time
import unittest

from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_pair_capture import ProcessPairCapture
from embodied_agent.scheduled_pair_capture import (
    ScheduledPairReader,
    ScheduledPairRequest,
    ScheduledReaderFactory,
)
from embodied_agent.threaded_pair_capture import CapturedPair
from scripts.check_process_pair_capture import fast_factory


class WaitMetricsTests(unittest.TestCase):
    def test_measurement_and_clear_on_invalid_clock(self):
        now = [0.0]

        def clock():
            now[0] += 0.0001
            return now[0]

        def sleep(delay):
            now[0] += delay

        def read(request):
            point = StampedPosition((0, 0, 0), now[0])
            return CapturedPair(request.session_id, point, point)

        reader = ScheduledPairReader(
            read, clock=clock, sleep=sleep, spin_window_s=0.001, measure_wait=True
        )
        reader(ScheduledPairRequest(1, "s", 0, 0.01, 0.012))
        metrics = reader.last_wait_metrics
        self.assertEqual(metrics["request_id"], 1)
        self.assertGreater(metrics["sleep_calls"], 0)
        self.assertGreater(metrics["spin_iterations"], 0)
        self.assertGreaterEqual(metrics["wait_cpu_s"], 0)
        self.assertFalse(metrics["window_missed"])
        reader.clock = lambda: float("nan")
        with self.assertRaises(ValueError):
            reader(ScheduledPairRequest(2, "s", 0, 0.01, 0.012))
        self.assertIsNone(reader.last_wait_metrics)

    def test_failed_window_metrics_cross_process(self):
        backend = ProcessPairCapture(
            ScheduledReaderFactory(fast_factory, measure_wait=True), capacity=1, trace_timing=True
        )
        try:
            self.assertTrue(backend.wait_ready())
            now = time.monotonic()
            backend.submit(ScheduledPairRequest(1, "s", now - 0.1, now - 0.05, now - 0.048))
            deadline = time.monotonic() + 2
            packets = []
            while not packets and time.monotonic() < deadline:
                packets = backend.drain()
                time.sleep(0.001)
            self.assertIn("window_missed", packets[0][2])
            row = backend.result()["timing_trace"][0]["reservation_wait"]
            self.assertTrue(row["window_missed"])
            self.assertEqual(row["request_id"], 1)
        finally:
            backend.cancel()
            backend.reap(0.25)

    def test_disabled_metrics_and_bad_option(self):
        reader = ScheduledPairReader(lambda _: None)
        self.assertIsNone(reader.last_wait_metrics)
        with self.assertRaises(ValueError):
            ScheduledPairReader(lambda _: None, measure_wait=1)


if __name__ == "__main__":
    unittest.main()
