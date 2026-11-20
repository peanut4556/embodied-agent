import unittest

from embodied_agent.mock import MockPolicy, MockRobot
from embodied_agent.models import Action, Observation, Plan, PlanStep
from embodied_agent.planner import RuleBasedPlanner
from embodied_agent.runtime import AgentRuntime


class SilentRobot(MockRobot):
    def __init__(self, ignored):
        super().__init__()
        self.ignored = ignored
        self.calls = []

    def execute(self, action):
        self.calls.append(action.name)
        if action.name != self.ignored:
            super().execute(action)


class ConfirmationTests(unittest.TestCase):
    def test_silent_pick_failure_blocks_place(self):
        robot = SilentRobot("pick")
        result = AgentRuntime(RuleBasedPlanner(), MockPolicy(), robot).run("red block box")
        self.assertFalse(result.success)
        self.assertEqual(result.completed_steps, 1)
        self.assertEqual(robot.calls, ["locate", "pick"])

    def test_silent_place_failure_blocks_verify(self):
        robot = SilentRobot("place")
        result = AgentRuntime(RuleBasedPlanner(), MockPolicy(), robot).run("red block box")
        self.assertFalse(result.success)
        self.assertEqual(result.completed_steps, 2)
        self.assertEqual(robot.calls, ["locate", "pick", "place"])

    def test_policy_destination_substitution_never_executed(self):
        class Policy(MockPolicy):
            def select_action(self, step, observation):
                action = super().select_action(step, observation)
                if action.name == "place":
                    return Action("place", {"object": "red_block", "destination": "elsewhere"})
                return action

        robot = SilentRobot("none")
        result = AgentRuntime(RuleBasedPlanner(), Policy(), robot).run("red block box")
        self.assertFalse(result.success)
        self.assertEqual(robot.calls, ["locate", "pick"])

    def test_unobservable_target_never_picked(self):
        robot = SilentRobot("none")
        robot._observation = Observation()
        result = AgentRuntime(RuleBasedPlanner(), MockPolicy(), robot).run("red block box")
        self.assertFalse(result.success)
        self.assertEqual(robot.calls, ["locate"])

    def test_stop_not_reported_as_success(self):
        class Planner:
            def create_plan(self, *args):
                return Plan("stop", [PlanStep("stop")])

        result = AgentRuntime(Planner(), MockPolicy(), MockRobot()).run("stop")
        self.assertFalse(result.success)
        self.assertEqual(result.message, "task stopped")


if __name__ == "__main__":
    unittest.main()
