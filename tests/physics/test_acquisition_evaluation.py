import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_acquisition_recovery import comparisons, design


class AcquisitionEvaluationTests(unittest.TestCase):
    def test_grid_and_frozen_deadline(self):
        config = json.loads(Path("config/acquisition-recovery.json").read_text())
        self.assertEqual(len(design(config)), 16)
        self.assertFalse(set(config["positions"]) & set(config["excluded_positions"]))
        for key, value in [
            ("acquisition_seconds", 3),
            ("modes", ["policy", "acquisition"]),
            ("positions", [0.283]),
        ]:
            bad = copy.deepcopy(config)
            bad[key] = value
            with self.assertRaises(ValueError):
                design(bad)

    def test_comparisons_separate_rescues_regressions_and_unnecessary_takeovers(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trace = np.zeros((4, 5))
            np.savez(output / "trace.npz", actions=trace, qpos=trace, features=trace)
            rows = []
            for case, results in [("a", [False, False, True]), ("b", [True, True, False])]:
                for mode, success in zip(
                    ("policy", "guarded", "acquisition"), results, strict=True
                ):
                    rows.append(
                        {
                            "case_id": case,
                            "mode": mode,
                            "visual": "nominal",
                            "trace": "trace.npz",
                            "initial_state": {},
                            "initial_image_sha256": "same",
                            "takeover_tick": None if mode == "policy" else 2,
                            "quality": {"verified_pick_place": success},
                            "failure": "none" if success else "no_stable_grasp",
                        }
                    )
            result = comparisons(rows, output)
            self.assertEqual(result["acquisition"]["rescued_vs_policy"], 1)
            self.assertEqual(result["acquisition"]["baseline_success_lost"], 1)
            self.assertEqual(result["acquisition"]["unnecessary_takeovers"], 1)
            self.assertEqual(result["guarded"]["rescued_vs_policy"], 0)
            self.assertEqual(result["policy"]["all"]["successes"], 1)
            self.assertTrue(
                all(c["equal_pre_takeover_prefix"] for c in result["acquisition"]["checks"])
            )
