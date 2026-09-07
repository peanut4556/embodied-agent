"""Optional MuJoCo tests; run make physics-test in the physics-capable environment."""

import unittest

import mujoco
import numpy as np

from embodied_agent.physics import PhysicsWorld


def execute(world, name, parameters):
    world.begin(name, parameters)
    for _ in range(300):
        if world.tick(0.04):
            return
    raise AssertionError("trajectory timed out")


class PhysicsTests(unittest.TestCase):
    def test_gravity_moves_free_cube_and_table_supports_it(self):
        world = PhysicsWorld()
        world.data.qpos[7] = 0.20  # cube z (freejoint starts at qpos[5])
        mujoco.mj_forward(world.model, world.data)
        world.tick(0.10)
        z = world.data.body("red_block").xpos[2]
        self.assertLess(z, 0.19)
        self.assertGreater(z, 0.05)
        world.tick(1.0)
        self.assertAlmostEqual(world.data.body("red_block").xpos[2], 0.025, places=3)

    def test_contact_grasp_then_release_into_tray(self):
        world = PhysicsWorld()
        execute(world, "pick", {"object": "red_block"})
        self.assertFalse(world.error)
        self.assertEqual(world.contacts(), 2)
        self.assertGreater(world.data.body("red_block").xpos[2], 0.20)
        execute(world, "place", {"object": "red_block", "destination": "box"})
        self.assertFalse(world.error)
        self.assertTrue(world.inside_box())
        self.assertEqual(world.contacts(), 0)
        world.tick(1.0)
        self.assertTrue(world.inside_box())

    def test_zero_friction_cannot_lift_cube(self):
        world = PhysicsWorld()
        world.model.geom_friction[:] = 0
        execute(world, "pick", {"object": "red_block"})
        self.assertIn("grasp failed", world.error)
        self.assertLess(world.data.body("red_block").xpos[2], 0.05)
        self.assertFalse(world.holding)

    def test_stop_holds_actuators_but_physics_keeps_running(self):
        world = PhysicsWorld()
        world.begin("pick", {"object": "red_block"})
        world.tick(0.5)
        world.begin("stop", {})
        controls = world.data.ctrl.copy()
        old_time = world.data.time
        world.tick(0.5)
        np.testing.assert_array_equal(world.data.ctrl, controls)
        self.assertGreater(world.data.time, old_time)
        self.assertFalse(world.active)

    def test_reject_place_without_contact_and_premature_verify(self):
        world = PhysicsWorld()
        with self.assertRaisesRegex(ValueError, "contact grasp"):
            world.begin("place", {"object": "red_block", "destination": "box"})
        with self.assertRaisesRegex(ValueError, "verification failed"):
            world.begin("verify", {"object": "red_block"})
