import unittest
from itertools import pairwise

from embodied_agent.observation_schedule import observation_times_ms
from embodied_agent.settled_reference import SettledReferenceConfirmation
from scripts.evaluate_sampling_phase import force_at


class ObservationScheduleTests(unittest.TestCase):
    def test_budget_deadline_and_continuity(self):
        for mode, count in (("uniform_5hz", 16), ("phase_sweep", 16), ("uniform_20hz", 61)):
            times = observation_times_ms(mode)
            self.assertEqual(len(times), count)
            self.assertEqual((times[0], times[-1]), (0, 3000))
            self.assertTrue(all(0 < b - a <= 250 for a, b in pairwise(times)))
            self.assertTrue(all(t % 2 == 0 for t in times))

    def test_sweep_visits_multiple_phases(self):
        times = observation_times_ms("phase_sweep")
        self.assertEqual({t % 200 for t in times}, {0, 112, 122, 152, 160})
        self.assertEqual(times, observation_times_ms("phase_sweep"))
        self.assertIsInstance(times, tuple)

    def test_irregular_schedule_preserves_confirmation(self):
        check = SettledReferenceConfirmation(0, position_error_m=(0.001, 0.001, 0.001))
        first = None
        for ms in observation_times_ms("phase_sweep"):
            check.step(ms / 1000, (0.64, 0, 0.045), (0.64, 0, 0.045))
            if first is None and check.status == "confirmed_in_tray":
                first = ms
        self.assertEqual(first, 2400)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            observation_times_ms("unknown")
        with self.assertRaises(ValueError):
            force_at("phase_30", 0)

    def test_phase_grid_pulse_boundaries(self):
        for phase in (10, 50, 90, 130, 170):
            mode = f"phase_{phase}"
            for cycle in range(15):
                start = phase / 1000 + cycle * 0.2
                self.assertEqual(force_at(mode, start - 0.002), 0)
                self.assertEqual(force_at(mode, start), 1.2)
                self.assertEqual(force_at(mode, start + 0.018), 1.2)
                self.assertEqual(force_at(mode, start + 0.020), 0)


if __name__ == "__main__":
    unittest.main()
