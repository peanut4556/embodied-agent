import unittest

from embodied_agent.uncertainty_view import UncertaintyViewConfirmation
from scripts.evaluate_uncertainty_view import PROFILES, evaluate


class UncertaintyViewTests(unittest.TestCase):
    def test_selects_relevant_axis_and_acquires_only_next_tick(self):
        for axis, point in enumerate(((0.689, 0, 0.045), (0.64, 0.039, 0.045), (0.64, 0, 0.057))):
            check = UncertaintyViewConfirmation(0, PROFILES)
            calls = []

            def locate(profile, calls=calls, point=point):
                calls.append(profile)
                return point, point

            check.step(0, locate)
            check.step(0.2, locate)
            chosen = ("focus_x", "focus_y", "focus_z")[axis]
            self.assertEqual(check.profile, chosen)
            self.assertEqual(calls, ["coarse", "coarse"])
            self.assertEqual(check.status, "pending")
            self.assertIsNone(check.confirmation.rows[0]["accepted"])
            check.step(0.4, locate)
            self.assertEqual(calls[-1], chosen)

    def test_switch_requires_new_full_stability_window(self):
        check = UncertaintyViewConfirmation(0, PROFILES)
        locate = lambda profile: ((0.689, 0, 0.045),) * 2
        for tick in range(7):
            check.step(tick * 0.2, locate)
        self.assertEqual(check.status, "pending")
        check.step(1.4, locate)
        self.assertEqual(check.status, "confirmed_in_tray")

    def test_missing_disagreement_and_invalid_do_not_trigger_precision_switch(self):
        for pair in (
            (None, None),
            ((0.64, 0, 0.045), (0.68, 0, 0.045)),
            ((float("nan"), 0, 0), None),
        ):
            check = UncertaintyViewConfirmation(0, PROFILES)
            for tick in range(16):
                check.step(tick * 0.2, lambda profile, pair=pair: pair)
            self.assertEqual(check.switches, [])
            self.assertEqual(check.status, "unconfirmed")

    def test_missing_breaks_consecutive_rejection_count(self):
        check = UncertaintyViewConfirmation(0, PROFILES)
        point = (0.689, 0, 0.045)
        check.step(0, lambda profile, point=point: (point, point))
        check.step(0.2, lambda profile: (None, None))
        check.step(0.4, lambda profile, point=point: (point, point))
        self.assertEqual(check.switches, [])

    def test_two_uncertain_axes_without_joint_improvement_stays(self):
        check = UncertaintyViewConfirmation(0, PROFILES)
        for tick in range(2):
            check.step(tick * 0.2, lambda profile: ((0.689, 0.039, 0.045),) * 2)
        self.assertEqual(check.switches, [])

    def test_only_one_switch_even_when_new_pair_stays_uncertain(self):
        check = UncertaintyViewConfirmation(0, PROFILES)
        for tick in range(16):
            point = (0.689, 0, 0.045) if tick < 2 else (0.64, 0.039, 0.045)
            check.step(tick * 0.2, lambda profile, point=point: (point, point))
        self.assertEqual(len(check.switches), 1)
        self.assertEqual(check.profile, "focus_x")
        self.assertEqual(check.status, "unconfirmed")

    def test_missing_after_switch_preserves_original_deadline(self):
        check = UncertaintyViewConfirmation(10, PROFILES)
        calls = []

        def locate(profile):
            calls.append(profile)
            return ((0.689, 0, 0.045),) * 2 if profile == "coarse" else (None, None)

        for tick in range(16):
            check.step(10 + tick * 0.2, locate)
        self.assertEqual(check.status, "unconfirmed")
        self.assertEqual(len(calls), 15)
        self.assertEqual(check.confirmation.confirmation.start, 10)

    def test_late_uncertainty_cannot_restart_budget(self):
        check = UncertaintyViewConfirmation(0, PROFILES)
        check.step(2, lambda profile: ((0.689, 0, 0.045),) * 2)
        check.step(2.2, lambda profile: ((0.689, 0, 0.045),) * 2)
        self.assertEqual(check.switches, [])

    def test_stop_and_invalid_clock_never_acquire(self):
        check = UncertaintyViewConfirmation(0, PROFILES)

        def forbidden(profile):
            self.fail("acquisition must not run")

        for timestamp in (-1, float("nan")):
            with self.assertRaises(ValueError):
                check.step(timestamp, forbidden)
        self.assertEqual(check.rows, [])
        self.assertEqual(check.step(0, forbidden, stop=True), "cancelled")
        check.step(0.2, forbidden)

    def test_bad_profiles_and_snapshot_isolation(self):
        for profiles in ({}, {"coarse": None}, {"coarse": (1, 2)}, {"coarse": (-1, 0, 0)}):
            with self.assertRaises(ValueError):
                UncertaintyViewConfirmation(0, profiles)
        check = UncertaintyViewConfirmation(0, PROFILES)
        check.step(0, lambda profile: ((0.689, 0, 0.045),) * 2)
        snapshot = check.result()
        snapshot["observations"][0]["observation"]["containment_margins_m"][0] = 10
        self.assertLess(check.rows[0]["observation"]["containment_margins_m"][0], 0)

    def test_acquisition_value_error_breaks_evidence(self):
        check = UncertaintyViewConfirmation(0, PROFILES)

        def fail(profile):
            raise ValueError("unusable capture")

        check.step(0, fail)
        self.assertEqual(check.rows[0]["error"], "unusable capture")
        self.assertEqual(check.confirmation.rows[0]["gate"], "missing_or_invalid_view")

    def test_experiment_has_gain_and_false_precision_counterexample(self):
        report = evaluate()
        self.assertEqual(len(report["cases"]), 162)
        self.assertTrue(report["matches_expected"])
