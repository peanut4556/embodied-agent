"""Force schedule boundaries and sub-frame audit regression."""

import unittest

from scripts.evaluate_force_disturbance import force_at, motion_summary


class ForceDisturbanceTests(unittest.TestCase):
    def test_single_pulses_half_open(self):
        for mode, onset in (("early_impulse", 0.4), ("late_impulse", 1.6)):
            self.assertEqual(force_at(mode, onset - 0.002), 0)
            self.assertEqual(force_at(mode, onset), 2)
            self.assertEqual(force_at(mode, onset + 0.018), 2)
            self.assertEqual(force_at(mode, onset + 0.02), 0)

    def test_interframe_force_is_invisible_at_capture_times(self):
        for i in range(15):
            self.assertEqual(force_at("interframe_impulses", i * 0.2), 0)
            self.assertEqual(force_at("interframe_impulses", i * 0.2 + 0.05), 1.2)
            self.assertEqual(force_at("interframe_impulses", i * 0.2 + 0.07), 0)
        self.assertEqual(force_at("interframe_impulses", 3.05), 0)

    def test_stationary_and_invalid(self):
        self.assertEqual(force_at("stationary", 1.5), 0)
        with self.assertRaises(ValueError):
            force_at("unknown", 0)

    def test_audit_keeps_between_frame_excursion(self):
        rows = [
            {"elapsed_s": t, "xyz": [x, 0, 0], "speed_m_s": v}
            for t, x, v in ((0, 5, 9), (1, 0, 0), (1.05, 0.003, 0.1), (1.2, 0, 0), (2, 0, 0))
        ]
        result = motion_summary(rows, 2)
        self.assertEqual(result["physics_samples"], 4)
        self.assertEqual(result["peak_speed_m_s"], 0.1)
        self.assertEqual(result["max_axis_excursion_m"], 0.003)


if __name__ == "__main__":
    unittest.main()
