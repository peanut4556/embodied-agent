import unittest

from embodied_agent.temporal_reference import GuardedMedianPair, TemporalReferenceConfirmation

BUDGET = (0.0013, 0.0016, 0.00051)
POINT = (0.64, 0, 0.04)


class TemporalReferenceTests(unittest.TestCase):
    def test_full_new_window_and_original_deadline(self):
        check = TemporalReferenceConfirmation(0, position_error_m=BUDGET)
        for tick in range(7):
            check.step(tick * 0.2, POINT, POINT)
        self.assertEqual(check.status, "pending")
        self.assertEqual(check.step(1.4, POINT, POINT), "confirmed_in_tray")
        late = TemporalReferenceConfirmation(0, position_error_m=BUDGET)
        for tick in range(10, 16):
            late.step(tick * 0.2, POINT, POINT)
        self.assertEqual(late.status, "unconfirmed")

    def test_median_removes_one_spike_without_lookahead(self):
        filt = GuardedMedianPair(BUDGET)
        self.assertEqual(filt.observe(0, POINT, POINT), (None, None))
        self.assertEqual(filt.observe(0.2, (0.64, 0, 0.0408), POINT), (None, None))
        self.assertEqual(filt.observe(0.4, POINT, POINT), (POINT, POINT))

    def test_missing_and_outside_raw_frame_never_hidden(self):
        for bad in (None, (0.7, 0, 0.04), (float("nan"), 0, 0.04)):
            filt = GuardedMedianPair(BUDGET)
            for tick in range(3):
                filt.observe(tick * 0.2, POINT, POINT)
            self.assertEqual(filt.observe(0.6, bad, POINT), (None, None))
            self.assertEqual(filt.observe(0.8, POINT, POINT), (None, None))

    def test_fast_motion_not_hidden_by_majority_stable_points(self):
        check = TemporalReferenceConfirmation(0, position_error_m=BUDGET)
        for tick in range(16):
            point = (0.63, 0, 0.04) if tick % 3 == 0 else POINT
            check.step(tick * 0.2, point, point)
        self.assertEqual(check.status, "unconfirmed")

    def test_slow_drift_exceeding_span_restarts_window(self):
        filt = GuardedMedianPair(BUDGET)
        for tick in range(3):
            p = (0.64 + tick * 0.0008, 0, 0.04)
            out = filt.observe(tick * 0.2, p, p)
        self.assertEqual(out, (None, None))
        self.assertEqual(filt.rows[-1]["filter_gate"], "window_motion_or_dispersion")

    def test_gap_restarts_window(self):
        filt = GuardedMedianPair(BUDGET)
        for tick in range(3):
            filt.observe(tick * 0.2, POINT, POINT)
        self.assertEqual(filt.observe(1, POINT, POINT), (None, None))

    def test_clock_errors_do_not_mutate_history(self):
        filt = GuardedMedianPair(BUDGET)
        filt.observe(0, POINT, POINT)
        for time in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                filt.observe(time, POINT, POINT)
        self.assertEqual(len(filt.rows), 1)

    def test_stop_and_snapshot(self):
        check = TemporalReferenceConfirmation(0, position_error_m=BUDGET)
        check.step(0, POINT, POINT)
        result = check.result()
        result["filter"]["observations"].clear()
        self.assertEqual(len(check.filter.rows), 1)
        self.assertEqual(check.step(0.2, None, None, stop=True), "cancelled")

    def test_shared_bias_is_not_corrected(self):
        check = TemporalReferenceConfirmation(0, position_error_m=BUDGET)
        # Could describe a true x=0.692 with a shared -8 mm error.
        for tick in range(8):
            check.step(tick * 0.2, (0.684, 0, 0.04), (0.684, 0, 0.04))
        self.assertEqual(check.status, "confirmed_in_tray")
