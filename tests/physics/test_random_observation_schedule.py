import random
import unittest
from collections import Counter
from itertools import pairwise

from embodied_agent.observation_schedule import PHASE_SWEEP_INTERVALS_MS
from embodied_agent.random_observation_schedule import randomized_observation_times_ms
from embodied_agent.settled_reference import SettledReferenceConfirmation
from scripts.evaluate_random_sampling import force_at


class RandomScheduleTests(unittest.TestCase):
    def test_many_seeds_preserve_budget_and_intervals(self):
        for seed in range(100):
            times = randomized_observation_times_ms(seed)
            self.assertEqual((len(times), times[0], times[-1]), (16, 0, 3000))
            self.assertEqual(
                Counter(b - a for a, b in pairwise(times)), Counter(PHASE_SWEEP_INTERVALS_MS)
            )
            self.assertTrue(all(t % 2 == 0 for t in times))

    def test_reproducible_and_no_global_rng_mutation(self):
        before = random.getstate()
        a = randomized_observation_times_ms(71)
        self.assertEqual(a, randomized_observation_times_ms(71))
        self.assertNotEqual(a, randomized_observation_times_ms(193))
        self.assertEqual(before, random.getstate())
        self.assertIsInstance(a, tuple)

    def test_bad_seeds(self):
        for seed in (-1, True, 1.5, "71", None):
            with self.assertRaises(ValueError):
                randomized_observation_times_ms(seed)

    def test_static_confirmable_for_all_frozen_schedules(self):
        for seed in (71, 193, 509):
            check = SettledReferenceConfirmation(0, position_error_m=(0.001,) * 3)
            for ms in randomized_observation_times_ms(seed):
                check.step(ms / 1000, (0.64, 0, 0.045), (0.64, 0, 0.045))
            self.assertEqual(check.status, "confirmed_in_tray")

    def test_new_period_boundaries(self):
        for period, phase in ((200, 10), (160, 30), (240, 70), (320, 110)):
            for cycle in range(8):
                t = (phase + cycle * period) / 1000
                mode = f"period_{period}_phase_{phase}"
                self.assertEqual(force_at(mode, t), 1.2)
                self.assertEqual(force_at(mode, t + 0.018), 1.2)
                self.assertEqual(force_at(mode, t + 0.020), 0)
                self.assertEqual(force_at(mode, t - 0.002), 0)
        with self.assertRaises(ValueError):
            force_at("period_1_phase_0", 0)
