import unittest

from embodied_agent.mock import MockPolicy, MockRobot
from embodied_agent.models import Action
from embodied_agent.planner import RuleBasedPlanner
from embodied_agent.runtime import AgentRuntime
from embodied_agent.safety import SafetyGate


class RuntimeTests(unittest.TestCase):
    def test_pick_and_place_task_completes(self):
        robot = MockRobot()
        result = AgentRuntime(RuleBasedPlanner(), MockPolicy(), robot).run(
            "把桌上的红色积木放进盒子"
        )

        self.assertTrue(result.success)
        self.assertEqual(result.completed_steps, 4)
        self.assertEqual(robot.observe().objects["red_block"], "box")
        self.assertEqual(robot.observe().gripper_holding, "")

    def test_safety_gate_rejects_unknown_action(self):
        with self.assertRaisesRegex(ValueError, "not allowed"):
            SafetyGate().validate_action(Action("move_without_limits"))


if __name__ == "__main__":
    unittest.main()
