"""Calibration fit accuracy and low-residual model-mismatch counterexamples."""

import unittest

import numpy as np

from embodied_agent.reference_fit import fit_reference, predict


class ReferenceFitTests(unittest.TestCase):
    def setUp(self):
        self.rays = np.array(
            [[x, y, -1.0] for x in np.linspace(-0.25, 0.25, 21) for y in np.linspace(-0.2, 0.2, 21)]
        )
        self.planes = [
            (np.array([0.0, 0.0, 1.0]), 0.0),
            (np.array([1.0, 0.0, 1.0]) / np.sqrt(2), 0.4 / np.sqrt(2)),
            (np.array([0.0, 1.0, 1.0]) / np.sqrt(2), 0.0),
        ]
        self.truth = np.r_[[0.002, -0.003, 0.004], np.deg2rad([0.2, -0.15, 0.1]), 0.002]

    def test_noiseless_recovery(self):
        result = fit_reference(self.rays, self.planes, predict(self.rays, self.planes, self.truth))
        self.assertEqual(result["status"], "converged")
        np.testing.assert_allclose(result["parameters"], self.truth, atol=1e-8, rtol=0)

    def test_plane_offset_is_absorbed_as_camera_translation(self):
        offsets = np.array([0.001, -0.001, 0.001])
        actual = [(n, d + v) for (n, d), v in zip(self.planes, offsets, strict=True)]
        result = fit_reference(self.rays, self.planes, predict(self.rays, actual, self.truth))
        expected = self.truth.copy()
        expected[:3] -= np.linalg.solve(np.array([n for n, d in self.planes]), offsets)
        np.testing.assert_allclose(result["parameters"], expected, atol=1e-8, rtol=0)
        self.assertLess(result["residual_rmse_m"], 1e-9)
        self.assertGreater(np.linalg.norm(expected[:3] - self.truth[:3]), 0.002)
        self.assertFalse(result["calibration_certified"])

    def test_single_plane_is_unidentifiable(self):
        planes = self.planes[:1]
        result = fit_reference(self.rays, planes, predict(self.rays, planes, self.truth))
        self.assertEqual(result["status"], "unidentifiable")
        self.assertLess(result["rank"], 7)

    def test_smaller_patch_is_worse_conditioned(self):
        large = fit_reference(self.rays, self.planes, predict(self.rays, self.planes, self.truth))
        rays = self.rays.copy()
        rays[:, :2] *= 0.1
        small = fit_reference(rays, self.planes, predict(rays, self.planes, self.truth))
        self.assertGreater(small["scaled_condition_number"], large["scaled_condition_number"] * 5)

    def test_invalid_observations(self):
        with self.assertRaises(ValueError):
            fit_reference(self.rays, self.planes, np.full((3, len(self.rays)), np.nan))
