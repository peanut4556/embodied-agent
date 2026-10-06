"""Rendered yaw invariance with unchanged occlusion, clipping and tilt rejection."""

import unittest

import mujoco
import numpy as np

from embodied_agent.perception import locate_red_cube
from embodied_agent.physics import PhysicsWorld


class OrientedRGBDTests(unittest.TestCase):
    def setUp(self):
        self.world = PhysicsWorld(perception="rgbd", perception_size_mode="oriented")
        self.addCleanup(self.world.close)
        self.renderer = mujoco.Renderer(self.world.model, height=480, width=640)
        self.addCleanup(self.renderer.close)

    def observe(self, angle, axis=(0, 0, 1)):
        world = self.world
        world.reset()
        world.data.qpos[5:8] = [0.32, 0, 0.025 if axis == (0, 0, 1) else 0.08]
        mujoco.mju_axisAngle2Quat(
            world.data.qpos[8:12], np.array(axis, dtype=float), np.deg2rad(angle)
        )
        mujoco.mj_forward(world.model, world.data)
        self.renderer.update_scene(world.data, camera="perception")
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        cam = world.model.camera("perception").id
        return (
            rgb,
            depth,
            world.data.cam_xpos[cam],
            world.data.cam_xmat[cam],
            world.model.cam_fovy[cam],
        )

    def test_yaw_angles_localize_without_widening_size_bounds(self):
        for angle in (0, 15, 30, 45, 60, 75, 90):
            with self.subTest(angle=angle):
                args = self.observe(angle)
                result = locate_red_cube(*args, size_mode="oriented")
                np.testing.assert_allclose(result["xyz"], [0.32, 0, 0.025], atol=0.004)
                if angle == 45:
                    with self.assertRaisesRegex(ValueError, "size inconsistent"):
                        locate_red_cube(*args)

    def test_half_occluded_rotated_cube_rejected(self):
        for angle in (0, 30, 45):
            rgb, *rest = self.observe(angle)
            red = (
                (rgb[:, :, 0] > 70)
                & (rgb[:, :, 0] > 1.6 * rgb[:, :, 1])
                & (rgb[:, :, 0] > 1.4 * rgb[:, :, 2])
            )
            _, xs = np.nonzero(red)
            rgb[:, : int(np.median(xs)) + 1] = 0
            with self.assertRaises(ValueError):
                locate_red_cube(rgb, *rest, size_mode="oriented")

    def test_tilted_cube_still_rejected(self):
        for angle in (15, 25, 37, 45):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                locate_red_cube(*self.observe(angle, (1, 0, 0)), size_mode="oriented")

    def test_clipped_rotated_cube_still_rejected(self):
        rgb, depth, *calibration = self.observe(30)
        red = (
            (rgb[:, :, 0] > 70)
            & (rgb[:, :, 0] > 1.6 * rgb[:, :, 1])
            & (rgb[:, :, 0] > 1.4 * rgb[:, :, 2])
        )
        _, xs = np.nonzero(red)
        shift = -int(xs.min())
        with self.assertRaisesRegex(ValueError, "clipped"):
            locate_red_cube(
                np.roll(rgb, shift, axis=1),
                np.roll(depth, shift, axis=1),
                *calibration,
                size_mode="oriented",
            )
