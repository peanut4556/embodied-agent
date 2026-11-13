import math
import unittest

from embodied_agent.capture_timing import analyze_capture_timing


def trace():
    return {
        "request_id": 1,
        "request_host_at": 10.0001,
        "submitted_at": 10.0002,
        "reader_started_at": 10.0005,
        "primary_exposure_at": 10.001,
        "secondary_exposure_at": 10.0015,
        "reader_returned_at": 10.0065,
        "result_ready_at": 10.0066,
        "drained_at": 10.007,
    }


class CaptureTimingTests(unittest.TestCase):
    def test_late_return_is_not_late_exposure(self):
        r = analyze_capture_timing(trace(), scheduled_host_at=10)
        self.assertEqual(r["status"], "measured")
        self.assertTrue(r["exposure_window_met"])
        self.assertAlmostEqual(r["segments_s"]["exposure_to_reader_return"], 0.005)
        self.assertAlmostEqual(sum(r["segments_s"].values()), r["total_s"])
        self.assertAlmostEqual(r["drain_age_s"], 0.006)
        self.assertFalse(r["predicts_future_feasibility"])

    def test_late_poll_cannot_be_hidden_by_fast_reader(self):
        r = analyze_capture_timing(trace(), scheduled_host_at=9.995)
        self.assertFalse(r["exposure_window_met"])
        self.assertEqual(r["largest_pre_exposure_segment"], "caller_poll")
        self.assertLess(r["exposure_margin_s"], 0)

    def test_exposure_window_boundary(self):
        row = trace()
        row["secondary_exposure_at"] = 10.002
        self.assertTrue(analyze_capture_timing(row, scheduled_host_at=10)["exposure_window_met"])
        row["secondary_exposure_at"] += 0.000001
        self.assertFalse(analyze_capture_timing(row, scheduled_host_at=10)["exposure_window_met"])

    def test_cached_future_and_unordered_stamps_are_invalid(self):
        for field, value in (
            ("primary_exposure_at", 9.99),
            ("secondary_exposure_at", 11),
            ("drained_at", 10),
            ("reader_started_at", 9.9),
        ):
            row = trace() | {field: value}
            r = analyze_capture_timing(row, scheduled_host_at=10)
            self.assertEqual(r["status"], "invalid_timestamp_order")
            self.assertNotIn("segments_s", r)

    def test_incomplete_cancelled_trace_is_not_zero_latency(self):
        r = analyze_capture_timing({"request_id": 2}, scheduled_host_at=10)
        self.assertEqual(r["status"], "incomplete")
        self.assertIn("drained_at", r["missing"])
        self.assertNotIn("exposure_window_met", r)

    def test_bad_values_and_budget(self):
        for value in (True, math.nan, math.inf, "10"):
            r = analyze_capture_timing(trace() | {"submitted_at": value}, scheduled_host_at=10)
            self.assertEqual(r["status"], "invalid_timestamps")
        for value in (-1, True, math.nan, math.inf, "2"):
            with self.assertRaises(ValueError):
                analyze_capture_timing(trace(), scheduled_host_at=10, exposure_window_s=value)

    def test_reader_error_is_not_usable_measurement(self):
        r = analyze_capture_timing(trace() | {"error": "failure"}, scheduled_host_at=10)
        self.assertEqual(r["status"], "reader_error")


if __name__ == "__main__":
    unittest.main()
