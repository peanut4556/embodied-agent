import unittest
from unittest.mock import patch

from embodied_agent.observation_capture import StampedPosition
from embodied_agent.scheduled_pair_capture import (
    ScheduledPairReader,
    ScheduledPairRequest,
    ScheduledReaderFactory,
)
from embodied_agent.threaded_pair_capture import CapturedPair


class BudgetTests(unittest.TestCase):
    def reader(self, budget):
        self.t = 0.0

        def clock():
            self.t += 0.0001
            return self.t

        def sleep(delay):
            self.t += delay

        def read(req):
            point = StampedPosition((0, 0, 0), self.t)
            return CapturedPair(req.session_id, point, point)

        return ScheduledPairReader(
            read, clock=clock, sleep=sleep, spin_window_s=0.0015, wait_cpu_budget_s=budget
        )

    def request(self, identity=1, session="s"):
        return ScheduledPairRequest(identity, session, self.t, self.t + 0.01, self.t + 0.012)

    def test_budget_crossing_disables_next_request_even_new_session(self):
        reader = self.reader(0.001)
        with patch(
            "embodied_agent.scheduled_pair_capture.time.thread_time",
            side_effect=[0, 0.002, 0.002, 0.0021],
        ):
            reader(self.request())
            self.assertGreater(reader.last_wait_metrics["spin_iterations"], 0)
            self.assertTrue(reader.last_wait_metrics["budget_exhausted"])
            reader(self.request(2, "new"))
        self.assertEqual(reader.last_wait_metrics["effective_spin_window_s"], 0)
        self.assertEqual(reader.last_wait_metrics["spin_iterations"], 0)
        self.assertAlmostEqual(reader.wait_cpu_spent_s, 0.0021)

    def test_zero_budget_never_spins_and_measures_implicitly(self):
        reader = self.reader(0)
        reader(self.request())
        self.assertEqual(reader.last_wait_metrics["spin_iterations"], 0)
        self.assertTrue(reader.last_wait_metrics["budget_exhausted"])

    def test_failed_clock_still_charged_and_metrics_cleared(self):
        reader = self.reader(0.001)
        reader.clock = lambda: float("nan")
        with (
            patch("embodied_agent.scheduled_pair_capture.time.thread_time", side_effect=[0, 0.002]),
            self.assertRaisesRegex(ValueError, "clock_invalid"),
        ):
            reader(self.request())
        self.assertAlmostEqual(reader.wait_cpu_spent_s, 0.002)
        self.assertIsNone(reader.last_wait_metrics)

    def test_expired_request_charged_without_read(self):
        reader = self.reader(0.001)
        reader.reader = lambda _: self.fail("expired request read")
        req = self.request()
        self.t = 0.02
        with (
            patch("embodied_agent.scheduled_pair_capture.time.thread_time", side_effect=[0, 0.002]),
            self.assertRaisesRegex(ValueError, "window_missed"),
        ):
            reader(req)
        self.assertTrue(reader.last_wait_metrics["budget_exhausted"])

    def test_invalid_budget_and_factory_forwarding(self):
        for budget in (-1, True, float("nan"), float("inf"), 1.1, "1"):
            with self.assertRaises(ValueError):
                self.reader(budget)
        reader = ScheduledReaderFactory(lambda: lambda _: None, 0.0015, False, 0.01)()
        self.assertEqual(reader.wait_cpu_budget_s, 0.01)
        self.assertTrue(reader.measure_wait)


if __name__ == "__main__":
    unittest.main()
