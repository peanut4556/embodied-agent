import copy
import unittest

from embodied_agent.empirical_bounds import estimate_profile_bounds


class EmpiricalBoundsTests(unittest.TestCase):
    def sample(self):
        return {
            "truth_xyz": (0.64, 0, 0.04),
            "primary": (0.642, -0.003, 0.041),
            "secondary": (0.636, 0.001, 0.038),
        }

    def test_both_views_and_axes_contribute(self):
        result = estimate_profile_bounds([self.sample()] * 6)
        for actual, expected in zip(result["position_error_m"], (0.0045, 0.0035, 0.0025)):
            self.assertAlmostEqual(actual, expected)
        self.assertFalse(result["certified"])
        self.assertEqual(result["complete_pairs"], 6)

    def test_default_requires_all_pairs(self):
        rows = [self.sample() for _ in range(7)]
        rows[-1]["secondary"] = None
        with self.assertRaises(ValueError):
            estimate_profile_bounds(rows)

    def test_explicit_coverage_records_missing_and_excludes_whole_pair(self):
        rows = [self.sample() for _ in range(6)]
        rows[0]["primary"] = None
        rows[0]["secondary"] = (1, 1, 1)
        rows[1]["secondary"] = None
        result = estimate_profile_bounds(rows, min_pairs=4, min_pair_fraction=2 / 3)
        self.assertEqual(result["missing_pair_indices"], [0, 1])
        self.assertEqual(result["complete_pairs"], 4)
        self.assertAlmostEqual(result["position_error_m"][0], 0.0045)

    def test_count_and_fraction_both_required(self):
        rows = [self.sample() for _ in range(6)]
        rows[0]["primary"] = None
        rows[1]["primary"] = None
        for kwargs in (
            {"min_pairs": 5, "min_pair_fraction": 0.5},
            {"min_pairs": 4, "min_pair_fraction": 0.8},
        ):
            with self.assertRaises(ValueError):
                estimate_profile_bounds(rows, **kwargs)

    def test_missing_is_not_zero_error(self):
        rows = [{"truth_xyz": (0, 0, 0), "primary": None, "secondary": None}] * 6
        with self.assertRaises(ValueError):
            estimate_profile_bounds(rows, min_pairs=1, min_pair_fraction=0.1)

    def test_invalid_positions_cannot_fit(self):
        for bad in ((float("nan"), 0, 0), (1, 2), (0, float("inf"), 0)):
            row = self.sample()
            row["primary"] = bad
            with self.assertRaises(ValueError):
                estimate_profile_bounds([row] * 6)

    def test_invalid_configuration(self):
        for kwargs in (
            {"padding_m": -1},
            {"padding_m": float("nan")},
            {"min_pairs": 0},
            {"min_pairs": True},
            {"min_pair_fraction": 0},
            {"min_pair_fraction": 1.1},
        ):
            with self.assertRaises(ValueError):
                estimate_profile_bounds([self.sample()] * 6, **kwargs)

    def test_frozen_results_do_not_alias_calibration(self):
        rows = [self.sample() for _ in range(6)]
        before = copy.deepcopy(rows)
        result = estimate_profile_bounds(rows)
        self.assertEqual(rows, before)
        rows[0]["primary"] = (1, 1, 1)
        self.assertAlmostEqual(result["position_error_m"][0], 0.0045)
