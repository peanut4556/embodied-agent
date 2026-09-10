"""Observation/action alignment at the physical integration boundary."""

import unittest
from types import SimpleNamespace

import numpy as np

from embodied_agent.physics import PhysicsWorld
from embodied_agent.policy_execution import PolicyMotion


class RecordingTests(unittest.TestCase):
    def learned_motion(self, world):
        motion = PolicyMotion.__new__(PolicyMotion)
        motion.world = world
        motion.period, motion.substeps, motion.elapsed = 0.04, 20, 0.0
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        motion.rgb = lambda: rgb
        target = world.data.ctrl.copy()
        target[0] += 0.01
        motion.controller = SimpleNamespace(state="running", reason="", step=lambda *args: target)
        return motion, target, rgb

    def test_learned_sample_contains_actual_input_and_new_action_before_integration(self):
        world = PhysicsWorld()
        motion, target, rgb = self.learned_motion(world)
        before, time = world.data.qpos.copy(), world.data.time
        captured = []

        def sample(current, image, holding):
            captured.append(
                (
                    current.data.time,
                    current.data.qpos.copy(),
                    current.data.ctrl.copy(),
                    image,
                    holding,
                )
            )

        motion.tick(0.04, sample)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], time)
        np.testing.assert_array_equal(captured[0][1], before)
        np.testing.assert_array_equal(captured[0][2], target)
        self.assertIs(captured[0][3], rgb)
        self.assertFalse(captured[0][4])
        self.assertGreater(world.data.time, time)
        world.close()

    def test_recorder_error_is_not_silently_relabelled_as_robot_failure(self):
        world = PhysicsWorld()
        motion, _, _ = self.learned_motion(world)
        before = world.data.time

        def broken(*args):
            raise ValueError("disk write failed")

        with self.assertRaisesRegex(ValueError, "disk write failed"):
            motion.tick(0.04, broken)
        self.assertEqual(world.data.time, before)
        world.close()

    def test_sample_precedes_motion_and_has_new_control_target(self):
        world = PhysicsWorld()
        initial_qpos = world.data.qpos.copy()
        initial_ctrl = world.data.ctrl.copy()
        initial_time = world.data.time
        world.begin("pick", {"object": "red_block"})
        samples = []

        def sample(current):
            samples.append((current.data.time, current.data.qpos.copy(), current.data.ctrl.copy()))

        world.tick(0.04, on_control_frame=sample)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0][0], initial_time)
        np.testing.assert_array_equal(samples[0][1], initial_qpos)
        self.assertFalse(np.array_equal(samples[0][2], initial_ctrl))
        self.assertGreater(world.data.time, samples[0][0])
