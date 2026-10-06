"""Static feasibility probes cannot mutate the live state or bypass arm limits."""

import unittest

import numpy as np

from embodied_agent.physics import PhysicsWorld
from scripts.audit_support_reach import probe


class SupportReachTests(unittest.TestCase):
    def test_probe_preserves_live_state_and_forward_kinematics(self):
        world = PhysicsWorld()
        try:
            before = [getattr(world.data, k).copy() for k in ("qpos", "qvel", "ctrl")]
            result = probe(world, 0.45, 0.30)
            self.assertTrue(result["reachable"])
            self.assertLess(result["fk_error_m"], 1e-10)
            self.assertAlmostEqual(result["tip_xyz"][1], 0)
            for key, expected in zip(("qpos", "qvel", "ctrl"), before, strict=True):
                np.testing.assert_array_equal(getattr(world.data, key), expected)
        finally:
            world.close()

    def test_outside_workspace_rejected(self):
        world = PhysicsWorld()
        try:
            self.assertFalse(probe(world, 2.0, 0.3)["reachable"])
        finally:
            world.close()

    def test_tray_collision_not_hidden_by_mount_classification(self):
        world = PhysicsWorld()
        try:
            result = probe(world, 0.545, 0.03)
            self.assertTrue(any("tray_left" in c["geoms"] for c in result["non_mount_collisions"]))
            self.assertTrue(any("pedestal" in c["geoms"] for c in result["collisions"]))
            self.assertFalse(any("pedestal" in c["geoms"] for c in result["non_mount_collisions"]))
        finally:
            world.close()
