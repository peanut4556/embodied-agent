import unittest

from embodied_agent.release_confirmation import ReleaseConfirmation
from scripts.evaluate_release_confirmation import observation_missing


class ReleaseScheduleTests(unittest.TestCase):
    def test_missing_schedules_cannot_accumulate_stale_positions(self):
        for mode, expected in [
            (True, "unconfirmed"),
            ("intermittent", "unconfirmed"),
            ("early", "confirmed_in_tray"),
        ]:
            with self.subTest(mode=mode):
                check = ReleaseConfirmation(3)
                for i in range(15, 31):
                    point = None if observation_missing(mode, i) else [0.64, 0, 0.04]
                    check.observe(i / 5, point)
                self.assertEqual(check.status, expected)

    def test_early_schedule_boundary_and_unknown_mode(self):
        self.assertTrue(observation_missing("early", 17))
        self.assertFalse(observation_missing("early", 18))
        self.assertFalse(observation_missing(False, 18))
        with self.assertRaises(ValueError):
            observation_missing("typo", 0)
