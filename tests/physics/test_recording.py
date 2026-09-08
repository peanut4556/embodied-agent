"""Observation/action alignment at the physical integration boundary."""

import unittest

import numpy as np

from embodied_agent.physics import PhysicsWorld


class RecordingTests(unittest.TestCase):
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
