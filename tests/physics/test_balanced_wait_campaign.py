import copy
import unittest
from collections import Counter
from itertools import pairwise

from scripts.check_balanced_wait_campaign import analyze, plan


def fixture():
    cases = []
    for p in plan():
        events = [{"event": "ready", "pid": i} for i in (1, 2)]
        events += [{"event": "stopped", "pid": i, "reason": "requested"} for i in (1, 2)]
        cases.append(
            {
                "plan": p,
                "spin_window_s": p["spin_window_s"],
                "rows": [{"request_id": i, "outcome": "delivered"} for i in range(1, 37)],
                "capture": {
                    "workers_stopped": True,
                    "timing_trace": [
                        {
                            "request_id": i,
                            "reservation_wait": {
                                "request_id": i,
                                "wait_cpu_s": 0.001 if p["spin_window_s"] else 0.0001,
                                "target_lateness_s": 0,
                            },
                        }
                        for i in range(1, 37)
                    ],
                },
                "load": {"events": events, "exitcodes": [0, 0]} if p["loaded"] else None,
            }
        )
    return cases


class CampaignTests(unittest.TestCase):
    def test_position_and_predecessor_balance(self):
        conditions = lambda p: (p["loaded"], p["spin_window_s"])
        for position in range(1, 5):
            self.assertEqual(len({conditions(p) for p in plan() if p["position"] == position}), 4)
        pairs = Counter()
        for block in range(1, 5):
            order = [conditions(p) for p in plan() if p["block"] == block]
            pairs.update(pairwise(order))
        self.assertEqual(len(pairs), 12)
        self.assertEqual(set(pairs.values()), {1})

    def test_failure_kept_and_paired_difference(self):
        cases = fixture()
        cases[0]["rows"][0]["outcome"] = "reader_error"
        s = analyze(cases)
        self.assertTrue(s["campaign_valid"])
        self.assertEqual(s["groups"][0]["outcomes"]["delivered"], 143)
        self.assertEqual(s["paired_blocks"][0]["spin_minus_sleep_delivered"], 1)
        self.assertFalse(s["automatic_enable"])

    def test_missing_reordered_or_mismatched_plan_rejected(self):
        cases = fixture()
        for bad in (cases[:-1], list(reversed(cases))):
            with self.assertRaises(ValueError):
                analyze(bad)
        cases[0]["spin_window_s"] = 0.0015
        with self.assertRaises(ValueError):
            analyze(cases)

    def test_incomplete_or_unclean_campaign_suppresses_comparisons(self):
        for fault in ("trace", "cleanup", "expired", "identity", "exit"):
            cases = copy.deepcopy(fixture())
            if fault == "trace":
                cases[0]["capture"]["timing_trace"].pop()
            elif fault == "cleanup":
                cases[0]["capture"]["workers_stopped"] = False
            elif fault == "expired":
                cases[2]["load"]["events"][-1]["reason"] = "lifetime"
            elif fault == "identity":
                cases[2]["load"]["events"][-1]["pid"] = 1
            else:
                cases[2]["load"]["exitcodes"] = [0, -9]
            result = analyze(cases)
            self.assertFalse(result["campaign_valid"])
            self.assertEqual(result["paired_blocks"], [])
            self.assertTrue(all(g["mean_wait_cpu_s"] is None for g in result["groups"]))


if __name__ == "__main__":
    unittest.main()
