"""A fourth plane reveals inconsistent locations but cannot anchor global translation."""

import unittest

import numpy as np

from embodied_agent.redundant_reference import check_redundant_reference
from embodied_agent.reference_fit import predict


class RedundantReferenceTests(unittest.TestCase):
    def setUp(self):
        self.rays = np.array(
            [[x, y, -1.0] for x in np.linspace(-0.25, 0.25, 21) for y in np.linspace(-0.2, 0.2, 21)]
        )
        self.planes = [
            (np.array([0.0, 0.0, 1.0]), 0.0),
            (np.array([1.0, 0.0, 1.0]) / np.sqrt(2), 0.4 / np.sqrt(2)),
            (np.array([0.0, 1.0, 1.0]) / np.sqrt(2), 0.0),
            (np.array([-1.0, 1.0, 1.0]) / np.sqrt(3), -0.4 / np.sqrt(3)),
        ]
        self.truth = np.r_[[0.002, -0.003, 0.004], np.deg2rad([0.2, -0.15, 0.1]), 0.002]

    def test_control_and_fourth_error_without_fit_leakage(self):
        clean = predict(self.rays, self.planes, self.truth)
        first = check_redundant_reference(self.rays, self.planes, clean)
        altered = clean.copy()
        altered[3] += 0.006
        second = check_redundant_reference(self.rays, self.planes, altered)
        self.assertEqual(first["status"], "no_alarm")
        self.assertEqual(second["status"], "alarm")
        self.assertEqual(first["fit"], second["fit"])

    def test_inconsistent_first_three_detected(self):
        actual = [
            (n, d + v) for (n, d), v in zip(self.planes, [0.001, -0.001, 0.001, 0], strict=True)
        ]
        result = check_redundant_reference(
            self.rays, self.planes, predict(self.rays, actual, self.truth)
        )
        self.assertEqual(result["status"], "alarm")

    def test_rigid_shift_remains_invisible(self):
        shift = np.array([0.001, -0.001, 0.002])
        actual = [(n, d + n @ shift) for n, d in self.planes]
        result = check_redundant_reference(
            self.rays, self.planes, predict(self.rays, actual, self.truth)
        )
        self.assertEqual(result["status"], "no_alarm")
        np.testing.assert_allclose(
            np.array(result["fit"]["parameters"])[:3], self.truth[:3] - shift, atol=1e-8
        )
        self.assertFalse(result["calibration_certified"])

    def test_unidentifiable_training_fit_stays_unknown(self):
        planes = [self.planes[0]] * 3 + [self.planes[3]]
        result = check_redundant_reference(
            self.rays, planes, predict(self.rays, planes, self.truth)
        )
        self.assertEqual(result["status"], "unknown")

    def test_invalid_observations(self):
        with self.assertRaises(ValueError):
            check_redundant_reference(self.rays, self.planes, np.zeros((3, len(self.rays))))
