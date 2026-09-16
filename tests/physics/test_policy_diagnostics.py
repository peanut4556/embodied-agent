"""Divergence timing and non-interfering optional physics traces."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from embodied_agent.physics import PhysicsWorld
from embodied_agent.reactive_evaluation import run_case
from scripts.diagnose_correction_policy import first_sustained


class DiagnosticsTests(unittest.TestCase):
    def test_first_sustained_reports_start_not_confirmation_tick(self):
        self.assertEqual(first_sustained([False, True, True, True], 3), 1)
        self.assertEqual(first_sustained([True] * 5), 0)

    def test_isolated_errors_and_short_sequences_do_not_trigger(self):
        self.assertIsNone(first_sustained([True, False, True, True], 3))
        self.assertIsNone(first_sustained([]))

    def test_trace_does_not_change_commands_or_physical_outcome(self):
        world = PhysicsWorld()
        try:
            bounds = world.model.actuator_ctrlrange.copy()
            command = world.data.qpos[:5].copy()
        finally:
            world.close()
        session = SimpleNamespace(
            metadata={"fps": 25, "history": 1},
            bounds=bounds,
            predict=lambda observation: command[None].copy(),
        )
        policy = SimpleNamespace(metadata={"fps": 25}, session=lambda: session)
        case = {"name": "trace-stop", "x": 0.312, "stop_at": 0.2}
        with patch("embodied_agent.reactive_evaluation.mujoco.Renderer") as renderer:
            renderer.return_value.render.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
            plain, _ = run_case(policy, case, "memory", 1)
            trace = []
            traced, _ = run_case(policy, case, "memory", 1, trace=trace)
        self.assertEqual(plain, traced)
        self.assertTrue(trace)
        self.assertEqual([r["tick"] for r in trace], list(range(len(trace))))
        self.assertTrue(traced["user_stop_passed"])
        for row in trace[5:]:
            np.testing.assert_array_equal(row["command"], trace[5]["command"])
