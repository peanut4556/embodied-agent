"""Necessary geometry contradictions never turn missing faces into a certificate."""

import unittest

import numpy as np

from embodied_agent.cube_consistency import cube_consistency


class CubeConsistencyTests(unittest.TestCase):
    def test_rotated_noisy_cube_diameter_is_allowed(self):
        rng = np.random.default_rng(73)
        for _ in range(30):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            points = rng.uniform(-0.025, 0.025, (50, 3)) @ rotation.T
            points += rng.uniform(-0.003, 0.003, points.shape)
            result = cube_consistency(points, [])
            self.assertFalse(result["size_contradiction"])
            self.assertFalse(result["clearance_certified"])

    def test_size_contradiction_detected(self):
        points = np.zeros((30, 3))
        points[-1, 0] = 0.11
        self.assertTrue(cube_consistency(points, [])["size_contradiction"])

    def test_orthogonal_parallel_and_skew_relations(self):
        points = np.zeros((30, 3))

        def groups(normal):
            return [
                {"indices": list(range(15)), "normal": [1, 0, 0]},
                {"indices": list(range(15, 30)), "normal": normal},
            ]

        for normal in ([1, 0, 0], [0, 1, 0]):
            self.assertFalse(cube_consistency(points, groups(normal))["relation_alarm"])
        self.assertTrue(
            cube_consistency(points, groups([np.sqrt(0.5), np.sqrt(0.5), 0]))["relation_alarm"]
        )

    def test_single_face_and_occlusion_are_not_confirmation(self):
        points = np.zeros((30, 3))
        result = cube_consistency(points, [{"indices": list(range(30)), "normal": [0, 0, 1]}])
        self.assertEqual(result["multiple_face_evidence"], "insufficient")
        self.assertEqual(cube_consistency(np.empty((0, 3)), [])["status"], "unknown")

    def test_same_face_diagonal_is_only_conditional_alarm(self):
        points = np.zeros((30, 3))
        points[-1, 0] = 0.085
        result = cube_consistency(points, [{"indices": list(range(30)), "normal": [0, 0, 1]}])
        self.assertFalse(result["size_contradiction"])
        self.assertTrue(result["face_size_alarm"])

    def test_invalid_inputs(self):
        for points in (np.ones((30, 2)), np.full((30, 3), np.nan)):
            with self.assertRaises(ValueError):
                cube_consistency(points, [])
