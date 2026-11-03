import unittest

from embodied_agent.cross_view import CrossViewConfirmation


class CrossViewTests(unittest.TestCase):
    def test_agreement_requires_full_stable_window(self):
        check = CrossViewConfirmation(0)
        for tick in range(5):
            self.assertEqual(check.step(tick * 0.2, (0.64, 0, 0.04), (0.642, 0, 0.04)), "pending")
        self.assertEqual(check.step(1, (0.64, 0, 0.04), (0.642, 0, 0.04)), "confirmed_in_tray")

    def test_one_wrong_view_cannot_confirm(self):
        check = CrossViewConfirmation(0)
        for tick in range(16):
            check.step(tick * 0.2, (0.64, 0, 0.04), (0.8, 0, 0.04))
        self.assertEqual(check.status, "unconfirmed")
        self.assertTrue(all(row["gate"] == "position_disagreement" for row in check.rows))

    def test_small_distance_straddling_region_boundary_is_rejected(self):
        check = CrossViewConfirmation(0)
        check.step(0, (0.689, 0, 0.04), (0.691, 0, 0.04))
        self.assertEqual(check.rows[-1]["gate"], "region_disagreement")

    def test_rejection_breaks_stability_without_deadline_extension(self):
        for bad in (None, (float("nan"), 0, 0), (0.8, 0, 0.04)):
            check = CrossViewConfirmation(0)
            for tick in range(5):
                check.step(tick * 0.2, (0.64, 0, 0.04), (0.64, 0, 0.04))
            check.step(1, (0.64, 0, 0.04), bad)
            for tick in range(6, 11):
                check.step(tick * 0.2, (0.64, 0, 0.04), (0.64, 0, 0.04))
            self.assertEqual(check.status, "pending")
            check.step(3.2, (0.64, 0, 0.04), (0.64, 0, 0.04))
            self.assertNotEqual(check.status, "confirmed_in_tray")

    def test_shared_bias_is_explicitly_not_detectable_from_agreement(self):
        check = CrossViewConfirmation(0)
        for tick in range(6):
            check.step(tick * 0.2, (0.64, 0, 0.04), (0.64, 0, 0.04))
        self.assertEqual(check.status, "confirmed_in_tray")

    def test_stop_terminal_and_snapshot_isolation(self):
        check = CrossViewConfirmation(0)
        check.step(float("nan"), None, None, stop=True)
        self.assertEqual(check.status, "cancelled")
        check.step(1, (0.64, 0, 0.04), (0.64, 0, 0.04))
        snapshot = check.result()
        snapshot["observations"].append({})
        self.assertEqual(check.rows, [])

    def test_invalid_timestamp_does_not_append_evidence(self):
        check = CrossViewConfirmation(0)
        with self.assertRaises(ValueError):
            check.step(float("nan"), (0.64, 0, 0.04), (0.64, 0, 0.04))
        self.assertEqual(check.rows, [])

    def test_spatial_agreement_does_not_replace_motion_stability(self):
        check = CrossViewConfirmation(0)
        for tick in range(16):
            point = (0.63 if tick % 2 else 0.65, 0, 0.04)
            check.step(tick * 0.2, point, point)
        self.assertEqual(check.status, "unconfirmed")
        self.assertTrue(all(row["gate"] == "paired_agreement" for row in check.rows))
