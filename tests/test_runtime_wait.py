import unittest

from embodied_agent.mock import MockPolicy, MockRobot
from embodied_agent.models import Observation
from embodied_agent.planner import RuleBasedPlanner
from embodied_agent.runtime import AgentRuntime


class DelayedRobot(MockRobot):
    def __init__(self, delay=2):
        super().__init__()
        self.pending = None
        self.delay = delay
        self.reads = 0
        self.calls = []

    def execute(self, action):
        self.calls.append(action.name)
        self.pending = action
        self.reads = 0

    def observe(self):
        if self.pending:
            self.reads += 1
            if self.reads >= self.delay:
                super().execute(self.pending)
                self.pending = None
        return super().observe()


class WaitTests(unittest.TestCase):
    def runtime(self, robot, **kwargs):
        self.t = 0

        def sleep(delay):
            self.t += delay

        return AgentRuntime(
            RuleBasedPlanner(),
            MockPolicy(),
            robot,
            clock=lambda: self.t,
            sleep=sleep,
            confirmation_timeout_s=0.1,
            **kwargs,
        )

    def test_delayed_grasp_and_place_succeed_without_reexecution(self):
        robot = DelayedRobot()
        result = self.runtime(robot).run("red block box")
        self.assertTrue(result.success)
        self.assertEqual(robot.calls, ["locate", "pick", "place", "verify"])

    def test_timeout_stops_before_place(self):
        robot = DelayedRobot(100)
        result = self.runtime(robot).run("red block box")
        self.assertFalse(result.success)
        self.assertIn("timeout", result.message)
        self.assertEqual(robot.calls, ["locate", "pick"])

    def test_cancel_while_waiting_does_not_replay_action(self):
        robot = DelayedRobot(100)
        result = self.runtime(robot).run("red block box", stop_requested=lambda: self.t >= 0.02)
        self.assertIn("cancelled", result.message)
        self.assertEqual(robot.calls, ["locate", "pick"])

    def test_read_limit_and_clock_validation(self):
        robot = DelayedRobot(100)
        runtime = self.runtime(robot, max_confirmation_reads=2)
        result = runtime.run("red block box")
        self.assertIn("read_limit", result.message)
        runtime.clock = lambda: float("nan")
        with self.assertRaises(ValueError):
            runtime.run("red block box")

    def test_late_positive_observation_is_not_confirmed(self):
        robot = MockRobot()
        runtime = self.runtime(robot)

        def observe():
            self.t += 0.2
            return Observation(objects={"red_block": "table"})

        robot.observe = observe
        self.assertIn("timeout", runtime.run("red block box").message)

    def test_invalid_configuration(self):
        for kwargs in (
            {"confirmation_timeout_s": True},
            {"confirmation_timeout_s": 6},
            {"confirmation_poll_s": 0},
            {"max_confirmation_reads": False},
        ):
            with self.assertRaises(ValueError):
                AgentRuntime(RuleBasedPlanner(), MockPolicy(), MockRobot(), **kwargs)


if __name__ == "__main__":
    unittest.main()
