import unittest

from embodied_agent.settled_reference import SettledReferenceConfirmation
from embodied_agent.temporal_reference import TemporalReferenceConfirmation

BUDGET = (0.0013, 0.0016, 0.00051)


class SettledReferenceTests(unittest.TestCase):
    def check(self):
        return SettledReferenceConfirmation(0, position_error_m=BUDGET)

    def test_static_needs_two_seconds(self):
        c = self.check()
        for i in range(10):
            c.step(i * 0.2, (0.64, 0, 0.04), (0.64, 0, 0.04))
        self.assertEqual(c.status, "pending")
        self.assertEqual(c.step(2, (0.64, 0, 0.04), (0.64, 0, 0.04)), "confirmed_in_tray")

    def test_slow_drift_that_short_filter_accepts_is_rejected(self):
        old = TemporalReferenceConfirmation(0, position_error_m=BUDGET)
        c = self.check()
        for i in range(16):
            p = (0.64 + 0.002 * i * 0.2, 0, 0.04)
            old.step(i * 0.2, p, p)
            c.step(i * 0.2, p, p)
        self.assertEqual(old.status, "confirmed_in_tray")
        self.assertEqual(c.status, "unconfirmed")
        self.assertTrue(any(r["gate"] == "sustained_raw_drift" for r in c.rows))

    def test_secondary_drift_cannot_be_hidden(self):
        c = self.check()
        for i in range(16):
            c.step(i * 0.2, (0.64, 0, 0.04), (0.64 + 0.002 * i * 0.2, 0, 0.04))
        self.assertEqual(c.status, "unconfirmed")

    def test_motion_then_settling_can_confirm(self):
        c = self.check()
        for i in range(16):
            p = (0.64 + 0.002 * min(i * 0.2, 0.6), 0, 0.04)
            c.step(i * 0.2, p, p)
        self.assertEqual(c.status, "confirmed_in_tray")

    def test_missing_breaks_long_window_without_extending_deadline(self):
        c = self.check()
        for i in range(16):
            p = None if i == 7 else (0.64, 0, 0.04)
            c.step(i * 0.2, p, p)
        self.assertEqual(c.status, "unconfirmed")

    def test_stale_clock_does_not_mutate(self):
        c = self.check()
        c.step(0, (0.64, 0, 0.04), (0.64, 0, 0.04))
        with self.assertRaises(ValueError):
            c.step(0, (0.64, 0, 0.04), (0.64, 0, 0.04))
        self.assertEqual(len(c.rows), 1)

    def test_subthreshold_drift_is_not_claimed_to_be_zero(self):
        c = self.check()
        for i in range(16):
            p = (0.64 + 0.0004 * i * 0.2, 0, 0.04)
            c.step(i * 0.2, p, p)
        self.assertEqual(c.status, "confirmed_in_tray")

    def test_stop_and_snapshot(self):
        c = self.check()
        c.step(0, (0.64, 0, 0.04), (0.64, 0, 0.04))
        snapshot = c.result()
        snapshot["observations"].clear()
        self.assertEqual(len(c.rows), 1)
        self.assertEqual(c.step(0.2, None, None, stop=True), "cancelled")
