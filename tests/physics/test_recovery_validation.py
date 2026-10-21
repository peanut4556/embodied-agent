import copy
import json
import unittest
from pathlib import Path

from scripts.evaluate_recovery_positives import run
from scripts.evaluate_recovery_validation import summarize, validate


class RecoveryValidationTests(unittest.TestCase):
    def test_frozen_new_timing_values_do_not_overlap_development(self):
        config = json.loads(Path("config/recovery-validation.json").read_text())
        validate(config)
        self.assertEqual(len(config["stretches"]), 8)
        for values in ([], [1.25], [1.4, 1.4], [float("nan")], [True], [3.1]):
            bad = copy.deepcopy(config)
            bad["stretches"] = values
            with self.assertRaises(ValueError):
                validate(bad)
        config["modes"] = ["baseline", "descent"]
        with self.assertRaises(ValueError):
            validate(config)

    def test_summary_separates_unnecessary_rescued_and_lost_outcomes(self):
        rows = []
        for stretch, outcomes in [(1.1, [True, True, False]), (1.4, [False, False, True])]:
            for mode, success in zip(("baseline", "fixed", "descent"), outcomes, strict=True):
                rows.append(
                    {
                        "stretch": stretch,
                        "mode": mode,
                        "quality": {"verified_pick_place": success},
                        "takeover_tick": None if mode == "baseline" else 100,
                    }
                )
        scores = summarize(rows)
        self.assertEqual(scores["baseline"]["unnecessary_takeovers"], 0)
        self.assertEqual(scores["descent"]["baseline_successes"], 1)
        self.assertEqual(scores["descent"]["unnecessary_takeovers"], 1)
        self.assertEqual(scores["descent"]["rescued_failures"], 1)
        self.assertEqual(scores["descent"]["lost_successes"], 1)
        self.assertEqual(scores["fixed"]["rescued_failures"], 0)

    def test_invalid_stop_requests_fail_before_creating_world(self):
        for mode, tick in [("baseline", 10), ("descent", -1), ("descent", 725), ("descent", True)]:
            with self.assertRaises(ValueError):
                run(None, None, 1.4, mode, None, stop_tick=tick)
        with self.assertRaises(ValueError):
            run(None, None, 1.4, "unknown", None)
        with self.assertRaises(ValueError):
            run(None, None, 0, "baseline", None)
