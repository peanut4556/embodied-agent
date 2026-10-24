import unittest

from embodied_agent.release_confirmation import ReleaseConfirmation


class ReleaseConfirmationTests(unittest.TestCase):
    def test_stable_goal_requires_full_second(self):
        check = ReleaseConfirmation(0)
        for t in (0, 0.2, 0.4, 0.6, 0.8):
            self.assertEqual(check.observe(t, [0.64, 0, 0.04]), "pending")
        self.assertEqual(check.observe(1, [0.64, 0, 0.04]), "confirmed_in_tray")

    def test_missing_and_long_gap_reset_evidence(self):
        for missing in (True, False):
            check = ReleaseConfirmation(0)
            check.observe(0, [0.64, 0, 0.04])
            if missing:
                check.observe(0.2, None)
            self.assertEqual(check.observe(1, [0.64, 0, 0.04]), "pending")
            self.assertEqual(check.observe(3, None), "unconfirmed")

    def test_outside_airborne_and_late_observations_never_confirm(self):
        check = ReleaseConfirmation(0.3)
        self.assertEqual(check.observe(3.3 - 1e-12, None), "unconfirmed")
        for point in ([0.8, 0, 0.04], [0.64, 0, 0.15], None, [0.64, 0, 0.04]):
            check = ReleaseConfirmation(0)
            self.assertNotEqual(check.observe(3.1, point), "confirmed_in_tray")

    def test_motion_resets_and_invalid_inputs_rejected(self):
        check = ReleaseConfirmation(0)
        for i in range(16):
            status = check.observe(i / 5, [0.62 + 0.04 * (i % 2), 0, 0.04])
        self.assertEqual(status, "unconfirmed")
        for timestamp, point in ((3, None), (float("nan"), None), (4, [1, 2])):
            with self.assertRaises(ValueError):
                check.observe(timestamp, point)
