"""State-copy isolation and correctly timed action-chain diagnostics."""

import unittest

import mujoco
import numpy as np

from embodied_agent.physics import PhysicsWorld
from scripts.audit_action_chain import shadow_step, summarize


class ActionChainTests(unittest.TestCase):
    def test_shadow_preserves_live_state_and_matches_real_next_state(self):
        world = PhysicsWorld()
        try:
            world.tick(0.2)
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            snapshot = np.empty(mujoco.mj_stateSize(world.model, spec))
            mujoco.mj_getState(world.model, world.data, snapshot, spec)
            command = world.data.ctrl.copy()
            command[0] += 0.02
            predicted = shadow_step(world.model, world.data, command, 20)
            unchanged = np.empty_like(snapshot)
            mujoco.mj_getState(world.model, world.data, unchanged, spec)
            np.testing.assert_array_equal(snapshot, unchanged)
            world.data.ctrl[:] = command
            for _ in range(20):
                mujoco.mj_step(world.model, world.data)
            np.testing.assert_allclose(predicted, world.data.qpos[:5], atol=1e-12, rtol=0)
        finally:
            world.close()

    def test_summary_separates_rate_change_and_state_error(self):
        rows = []
        for tick in range(50):
            zeros = np.zeros(5)
            joints = zeros.copy()
            if tick >= 25:
                joints[0] = 0.2
            prediction = zeros.copy()
            prediction[0] = 0.1 if tick == 0 else 0
            rows.append(
                {
                    "tick": tick,
                    "prediction": prediction,
                    "command": zeros,
                    "joints_before": zeros,
                    "joints_after": joints,
                    "shadow_unlimited_next_joints": joints,
                    "reference_command": zeros,
                    "reference_next_joints": zeros,
                }
            )
        report = summarize(rows, 25)
        self.assertEqual(report["rate_limited_frames"], 1)
        self.assertEqual(report["first_sustained_next_state_error_tick"], 25)
        self.assertEqual(report["first_second"]["arm_state_rmse_rad"], 0)
        self.assertGreater(report["two_seconds"]["arm_state_rmse_rad"], 0)
        self.assertEqual(report["max_same_state_limit_effect_arm_rad"], 0)
