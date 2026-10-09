"""Plane proposals must tolerate bounded noise and reject underconstrained support."""

import unittest

import numpy as np

from embodied_agent.plane_groups import plane_groups


class PlaneGroupTests(unittest.TestCase):
    def test_two_faces_separate_with_stable_normals(self):
        rng = np.random.default_rng(11)
        a = np.c_[rng.uniform(-0.025, 0.025, (200, 2)), np.full(200, 0.025)]
        b = np.c_[np.full(160, 0.025), rng.uniform(-0.025, 0.025, (160, 2))]
        points = np.r_[a, b] + rng.uniform(-0.0001, 0.0001, (360, 3))
        result = plane_groups(points)
        groups = [g for g in result["groups"] if g["proposal_accepted"]]
        self.assertEqual(len(groups), 2)
        matched = []
        for group in groups:
            n = np.abs(group["normal"])
            matched.append(int(np.argmax(n)))
            self.assertGreater(n.max(), np.cos(np.deg2rad(1)))
        self.assertEqual(set(matched), {0, 2})
        assigned = [i for g in result["groups"] for i in g["indices"]]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(sorted(assigned + result["unassigned_indices"]), list(range(360)))
        self.assertEqual(result, plane_groups(points))
        self.assertFalse(result["angle_bound_certified"])

    def test_line_and_small_patch_are_not_accepted_planes(self):
        t = np.linspace(-0.02, 0.02, 80)
        line = np.c_[t, np.zeros(80), np.zeros(80)]
        tiny = np.random.default_rng(73).uniform(-0.001, 0.001, (80, 3))
        for points in (line, tiny):
            self.assertEqual(plane_groups(points)["status"], "unknown")

    def test_missing_and_invalid_evidence(self):
        for points in (np.empty((0, 3)), np.zeros((29, 3))):
            self.assertEqual(plane_groups(points)["status"], "unknown")
        for points in (np.zeros((30, 2)), np.full((30, 3), np.nan)):
            with self.assertRaises(ValueError):
                plane_groups(points)

    def test_rotated_noisy_plane(self):
        rng = np.random.default_rng(73)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        plane = np.c_[rng.uniform(-0.025, 0.025, (300, 2)), np.zeros(300)]
        points = plane @ rotation.T + rng.uniform(-0.0005, 0.0005, plane.shape)
        result = plane_groups(points)
        g = result["groups"][0]
        self.assertTrue(g["proposal_accepted"])
        self.assertGreater(abs(np.array(g["normal"]) @ rotation[:, 2]), np.cos(np.deg2rad(1)))
