import unittest

from embodied_agent.cross_view import CrossViewConfirmation
from scripts.evaluate_uncertainty_confirmation import evaluate


class UncertaintyConfirmationTests(unittest.TestCase):
    def check(self, bound=(0.004, 0.004, 0.004)):
        return CrossViewConfirmation(0, position_error_m=bound)

    def test_interior_confirms_after_full_window(self):
        check = self.check()
        for tick in range(6):
            check.step(tick * 0.2, (0.64, 0, 0.045), (0.641, 0, 0.045))
        self.assertEqual(check.status, "confirmed_in_tray")

    def test_all_six_faces_and_exact_contact_rejected(self):
        for point in (
            (0.594, 0, 0.045),
            (0.686, 0, 0.045),
            (0.64, -0.036, 0.045),
            (0.64, 0.036, 0.045),
            (0.64, 0, 0.036),
            (0.64, 0, 0.054),
        ):
            check = self.check()
            for tick in range(16):
                check.step(tick * 0.2, point, point)
            self.assertEqual(check.status, "unconfirmed")
            self.assertEqual(check.rows[0]["gate"], "uncertainty_overlaps_boundary")

    def test_secondary_box_also_must_fit(self):
        check = self.check()
        check.step(0, (0.68, 0, 0.045), (0.688, 0, 0.045))
        self.assertEqual(check.rows[0]["gate"], "uncertainty_overlaps_boundary")

    def test_anisotropic_bound_and_snapshot(self):
        bound = [0.001, 0.002, 0.003]
        check = self.check(bound)
        bound[0] = 1
        check.step(0, (0.688, 0, 0.045), (0.688, 0, 0.045))
        self.assertEqual(check.rows[0]["gate"], "paired_agreement")
        snapshot = check.result()
        snapshot["observations"][0]["containment_margins_m"][0] = -100
        self.assertGreater(check.rows[0]["containment_margins_m"][0], 0)

    def test_rejection_resets_window_and_preserves_deadline(self):
        check = self.check()
        for tick in range(16):
            point = (0.688, 0, 0.045) if tick in (4, 9, 14) else (0.684, 0, 0.045)
            check.step(tick * 0.2, point, point)
        self.assertEqual(check.status, "unconfirmed")

    def test_bad_bounds_rejected(self):
        for bound in (
            1,
            (),
            (1, 2),
            (1, 2, 3, 4),
            (-1, 0, 0),
            (0, float("nan"), 0),
            (0, 0, float("inf")),
        ):
            with self.assertRaises(ValueError):
                self.check(bound)

    def test_stop_and_invalid_clock(self):
        check = self.check()
        with self.assertRaises(ValueError):
            check.step(float("nan"), (0.64, 0, 0.045), (0.64, 0, 0.045))
        self.assertEqual(check.rows, [])
        self.assertEqual(check.step(0, None, None, stop=True), "cancelled")

    def test_rigid_grid_includes_underspecified_budget_counterexample(self):
        report = evaluate()
        self.assertEqual(len(report["cases"]), 108)
        self.assertTrue(report["matches_expected"])
        self.assertEqual(report["summary_by_budget_mm"]["4"]["guarded_false"], 0)
