import unittest

from embodied_agent.uncertainty_view import UncertaintyViewConfirmation

PROFILES = {"coarse": (0.004,) * 3, "detail": (0.001,) * 3}
GOOD = ((0.64, 0, 0.045),) * 2


class UncertaintyFallbackTests(unittest.TestCase):
    def check(self):
        return UncertaintyViewConfirmation(0, PROFILES, switch_on_missing=True)

    def test_missing_then_recovery_requires_full_new_window(self):
        check = self.check()
        calls = []

        def locate(profile):
            calls.append(profile)
            return (None, None) if profile == "coarse" else GOOD

        for tick in range(7):
            check.step(tick * 0.2, locate)
        self.assertEqual(check.status, "pending")
        self.assertEqual(calls[:3], ["coarse", "coarse", "detail"])
        self.assertEqual(check.switches[0]["reason"], "two_missing_pairs")
        check.step(1.4, locate)
        self.assertEqual(check.status, "confirmed_in_tray")

    def test_all_missing_stops_at_original_deadline(self):
        check = self.check()
        for tick in range(16):
            check.step(tick * 0.2, lambda profile: (None, None))
        self.assertEqual(check.status, "unconfirmed")
        self.assertEqual(len(check.switches), 1)

    def test_shared_single_switch_budget_across_reasons(self):
        check = self.check()
        for tick in range(16):
            check.step(
                tick * 0.2,
                lambda profile: ((0.6885, 0, 0.045),) * 2 if profile == "coarse" else (None, None),
            )
        self.assertEqual(len(check.switches), 1)
        self.assertEqual(check.switches[0]["reason"], "boundary_uncertainty")
        self.assertEqual(check.status, "unconfirmed")

    def test_disagreement_is_not_missing(self):
        check = self.check()
        for tick in range(16):
            check.step(tick * 0.2, lambda profile: ((0.64, 0, 0.045), (0.68, 0, 0.045)))
        self.assertEqual(check.switches, [])

    def test_valid_pair_breaks_missing_sequence(self):
        check = self.check()
        check.step(0, lambda profile: (None, None))
        check.step(0.2, lambda profile: GOOD)
        check.step(0.4, lambda profile: (None, None))
        self.assertEqual(check.switches, [])

    def test_no_alternative_and_late_missing_do_not_switch(self):
        for check, start in (
            (
                UncertaintyViewConfirmation(
                    0, {"coarse": PROFILES["coarse"]}, switch_on_missing=True
                ),
                0,
            ),
            (self.check(), 2),
        ):
            check.step(start, lambda profile: (None, None))
            check.step(start + 0.2, lambda profile: (None, None))
            self.assertEqual(check.switches, [])

    def test_profile_choice_uses_error_diagonal_and_deterministic_name(self):
        check = UncertaintyViewConfirmation(
            0, {**PROFILES, "a_detail": (0.001,) * 3}, switch_on_missing=True
        )
        check.step(0, lambda profile: (None, None))
        check.step(0.2, lambda profile: (None, None))
        self.assertEqual(check.profile, "a_detail")
        self.assertEqual(check.rows[-1]["predicted_candidate_margins_m"], {})

    def test_option_is_explicit_boolean(self):
        with self.assertRaises(ValueError):
            UncertaintyViewConfirmation(0, PROFILES, switch_on_missing="false")
