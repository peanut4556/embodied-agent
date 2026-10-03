"""Matched branching must include velocity/control and object targets use actual displacement."""

import copy
import unittest

import numpy as np

from scripts.audit_multistep_targets import target_row, verify_takeover


class MultistepTargetTests(unittest.TestCase):
    def test_takeover_requires_full_state_and_previous_command(self):
        record = {"qpos": [1.0, 2.0, 3.0], "qvel": [0.1, 0.2], "ctrl": [0.3, 0.4]}
        row = {
            "simulation_qpos": record["qpos"][:],
            "simulation_qvel": record["qvel"][:],
            "previous_command": record["ctrl"][:],
        }
        verify_takeover(row, record)
        for field in row:
            changed = copy.deepcopy(row)
            changed[field][0] += 0.001
            with self.assertRaisesRegex(ValueError, "takeover"):
                verify_takeover(changed, record)

    def test_object_target_is_separate_from_joint_error_and_preserves_split(self):
        expert = {
            "joints": [0.0] * 5,
            "block_xyz": [0.3, 0.0, 0.2],
            "holding": True,
            "inside_box": False,
        }
        model = {"joints": [0.0] * 5, "block_xyz": [0.3, 0.0, 0.0], "holding": False}
        row = target_row("validation", 8, 25, np.array([0.3, 0.0, 0.0]), expert, model)
        self.assertEqual(row["group"], "validation")
        self.assertEqual(row["horizon_ticks"], 25)
        self.assertEqual(row["arm_continuation_rmse_rad"], 0)
        self.assertAlmostEqual(row["object_target_displacement_m"], 0.2)
        self.assertAlmostEqual(row["object_continuation_error_m"], 0.2)
        self.assertTrue(row["expert_holding_target"])
        self.assertFalse(row["autonomous_holding"])

    def test_post_stop_trace_horizon_really_advances_physics(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from embodied_agent.physics import PhysicsWorld
        from embodied_agent.reactive_evaluation import run_case

        world = PhysicsWorld()
        try:
            bounds = world.model.actuator_ctrlrange.copy()
            action = world.data.qpos[:5].copy()
        finally:
            world.close()
        session = SimpleNamespace(
            metadata={"fps": 25, "history": 1}, bounds=bounds, predict=lambda _: action[None].copy()
        )
        policy = SimpleNamespace(metadata={"fps": 25}, session=lambda: session)
        with patch("embodied_agent.reactive_evaluation.mujoco.Renderer") as renderer:
            renderer.return_value.render.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
            trace = []
            result, _ = run_case(
                policy,
                {"name": "stop", "x": 0.312, "stop_at": 0.2},
                "memory",
                2,
                trace=trace,
                trace_until_tick=40,
            )
        self.assertEqual(len(trace), 41)
        self.assertAlmostEqual(result["duration_seconds"], 41 / 25)
        self.assertTrue(result["user_stop_passed"])
        for row in trace[6:]:
            np.testing.assert_array_equal(row["command"], trace[5]["command"])
        # Each frame was rendered/integrated; these are not copies of the final row.
        self.assertEqual(renderer.return_value.render.call_count, 41)
        self.assertEqual([r["tick"] for r in trace], list(range(41)))
        with self.assertRaisesRegex(ValueError, "trace horizon"):
            run_case(policy, {"name": "bad", "x": 0.312}, "memory", 2, trace_until_tick=100)
