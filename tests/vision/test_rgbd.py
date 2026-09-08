"""Rendered camera → localization → contact physics, requires OpenGL/OSMesa."""

import unittest

import mujoco
import numpy as np

from embodied_agent.physics import PhysicsWorld


def execute(world, name, parameters):
    world.begin(name, parameters)
    for _ in range(300):
        if world.tick(0.04):
            if world.error:
                raise AssertionError(world.error)
            return
    raise AssertionError("trajectory timeout")


class RGBDTests(unittest.TestCase):
    def setUp(self):
        self.world = PhysicsWorld(perception="rgbd")
        self.addCleanup(self.world.close)

    def test_shifted_cube_positions_drive_successful_grasps(self):
        world = self.world
        for x in (0.28, 0.32, 0.40):
            with self.subTest(x=x):
                world.reset()
                # Scene setup only. Ground truth is used exclusively for test scoring.
                world.data.qpos[5] = x
                mujoco.mj_forward(world.model, world.data)
                world.tick(0.2)
                detected = world.locate_block()
                np.testing.assert_allclose(detected, world.data.body("red_block").xpos, atol=0.004)
                execute(world, "pick", {"object": "red_block"})
                self.assertTrue(world.holding)
                execute(world, "place", {"object": "red_block", "destination": "box"})
                self.assertTrue(world.inside_box())

    def test_missing_red_target_does_not_reuse_previous_detection(self):
        world = self.world
        world.locate_block()
        world.model.geom("red_block").rgba[:] = [0.2, 0.2, 0.2, 1]
        with self.assertRaisesRegex(ValueError, "found 0"):
            world.begin("pick", {"object": "red_block"})
        self.assertIsNone(world.detection)
        self.assertFalse(world.active)

    def test_target_outside_motion_plane_is_rejected(self):
        world = self.world
        world.data.qpos[6] = 0.05
        mujoco.mj_forward(world.model, world.data)
        with self.assertRaisesRegex(ValueError, "not graspable"):
            world.begin("pick", {"object": "red_block"})
        self.assertFalse(world.active)
