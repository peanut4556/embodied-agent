import unittest

from embodied_agent.observation_capture import StampedPosition
from embodied_agent.scheduled_pair_capture import ScheduledPairReader, ScheduledPairRequest
from embodied_agent.threaded_pair_capture import CapturedPair


class SpinTests(unittest.TestCase):
    def test_spin_does_not_expose_early_or_sleep_in_final_window(self):
        now = [0.0]
        sleeps = []

        def clock():
            now[0] += 0.0001
            return now[0]

        def sleep(duration):
            sleeps.append((now[0], duration))
            now[0] += duration

        def read(request):
            self.assertGreaterEqual(now[0], request.target_host_time)
            point = StampedPosition((0, 0, 0), now[0])
            return CapturedPair(request.session_id, point, point)

        reader = ScheduledPairReader(read, clock=clock, sleep=sleep, spin_window_s=0.0015)
        reader(ScheduledPairRequest(1, "session", 0, 0.01, 0.012))
        self.assertTrue(sleeps)
        self.assertTrue(all(start + duration <= 0.0085 + 1e-9 for start, duration in sleeps))

    def test_oversleep_is_still_rejected(self):
        now = [0.0]

        def sleep(duration):
            now[0] = 0.02

        reader = ScheduledPairReader(
            lambda _: self.fail("reader must not run"),
            clock=lambda: now[0],
            sleep=sleep,
            spin_window_s=0.0015,
        )
        with self.assertRaisesRegex(ValueError, "window_missed"):
            reader(ScheduledPairRequest(1, "session", 0, 0.01, 0.012))

    def test_invalid_margin(self):
        for value in (True, -1, 0.003, float("nan")):
            with self.assertRaises(ValueError):
                ScheduledPairReader(lambda _: None, spin_window_s=value)


if __name__ == "__main__":
    unittest.main()
