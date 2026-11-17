import unittest

from scripts.check_fixed_wait_probe import summarize


class FixedProbeTests(unittest.TestCase):
    def test_failures_stay_in_denominator(self):
        rows = [
            {"request_id": 1, "outcome": "delivered"},
            {"request_id": 2, "outcome": "reader_error"},
        ]
        trace = [
            {
                "request_id": i,
                "reservation_wait": {
                    "request_id": i,
                    "wait_cpu_s": 0.001 * i,
                    "target_lateness_s": 0.001 * i,
                },
            }
            for i in (1, 2)
        ]
        result = summarize(rows, trace, 2)
        self.assertTrue(result["measurement_complete"])
        self.assertEqual(result["mean_wait_cpu_s"], 0.0015)
        self.assertEqual(result["outcomes"]["reader_error"], 1)

    def test_partial_measurement_does_not_publish_mean(self):
        rows = [
            {"request_id": 1, "outcome": "timeout"},
            {"request_id": 2, "outcome": "not_attempted"},
        ]
        result = summarize(rows, [], 2)
        self.assertFalse(result["measurement_complete"])
        self.assertIsNone(result["mean_wait_cpu_s"])

    def test_duplicate_and_missing_outcomes_rejected(self):
        for ids in ((1, 1), (1,), (1, 3)):
            with self.assertRaises(ValueError):
                summarize([{"request_id": i, "outcome": "delivered"} for i in ids], [], 2)

    def test_misassociated_trace_rejected(self):
        with self.assertRaises(ValueError):
            summarize(
                [{"request_id": 1, "outcome": "delivered"}],
                [{"request_id": 1, "reservation_wait": {"request_id": 2}}],
                1,
            )


if __name__ == "__main__":
    unittest.main()
