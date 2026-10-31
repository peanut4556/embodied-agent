import unittest
from itertools import product

from embodied_agent.release_confirmation import ReleaseConfirmation
from scripts.audit_stability_interruptions import POINTS, compare, evaluate, targeted_cases


class StabilityInterruptionTests(unittest.TestCase):
    def test_exhaustive_sequences_match_both_confirmation_paths(self):
        report = evaluate()
        self.assertEqual(report["exhaustive_cases"], 19683)
        self.assertEqual(report["targeted_cases"], 45)
        self.assertEqual(report["mismatches"], 0, report["exhaustive_failures"][:3])
        # A binary-exact 1 s window needs five consecutive in-region samples.
        expected = sum("IIIII" in "".join(word) for word in product("IOM", repeat=9))
        self.assertEqual(report["exhaustive_outcomes"]["confirmed"], expected)
        self.assertEqual(expected, 297)

    def test_preconfirmation_interruptions_restart_full_window(self):
        cases = dict(targeted_cases())
        for kind in ("outside", "missing", "jump"):
            for length in (1, 2):
                row = compare(cases[f"{kind}_2_{length}"])
                # Interrupt at .5; first stable point returns at .5 + length*.25.
                expected = 0.5 + length * 0.25 + 1
                self.assertEqual(row["actual_first_confirmation"], expected)
                self.assertEqual(row["dual_actual"], expected)

    def test_recovery_at_deadline_differs_from_delivered_evidence(self):
        cases = dict(targeted_cases())
        for kind in ("outside", "missing", "jump"):
            row = compare(cases[f"late_{kind}_7"])
            self.assertEqual(row["actual_first_confirmation"], 3)
            self.assertIsNone(row["dual_actual"])
            row = compare(cases[f"late_{kind}_8"])
            self.assertIsNone(row["actual_first_confirmation"])
            self.assertIsNone(row["dual_actual"])

    def test_gap_limit_resets_only_when_exceeded(self):
        cases = dict(targeted_cases())
        self.assertEqual(compare(cases["gap_0.25"])["actual_first_confirmation"], 1)
        self.assertAlmostEqual(
            compare(cases["gap_0.250001"])["actual_first_confirmation"], 1.750001
        )
        self.assertEqual(compare(cases["gap_0.5"])["actual_first_confirmation"], 2)

    def test_terminal_success_is_latched_not_continuous_monitoring(self):
        monitor = ReleaseConfirmation(0)
        for i in range(5):
            monitor.observe(i / 4, POINTS["I"])
        self.assertEqual(monitor.status, "confirmed_in_tray")
        monitor.observe(1.25, POINTS["O"])
        monitor.observe(1.5, None)
        self.assertEqual(monitor.status, "confirmed_in_tray")

    def test_fast_return_starts_from_new_position(self):
        samples = [(i / 4, POINTS["I"]) for i in range(9)]
        samples[3] = (0.75, (0.66, 0, 0.033))
        row = compare(samples)
        self.assertTrue(row["match"])
        self.assertEqual(row["actual_first_confirmation"], 2)
