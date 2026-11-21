import unittest

from embodied_agent.mock import MockPolicy
from embodied_agent.models import Observation, Plan, PlanStep
from embodied_agent.runtime import AgentRuntime


class Planner:
    def create_plan(self, *args):
        return Plan("verify", [PlanStep("verify", {"object": "block", "destination": "box"})])


class FreshnessTests(unittest.TestCase):
    def run_case(self, stamps, timeout=0.03, age=0.01):
        self.t = 10.0
        test = self

        class Robot:
            calls = 0

            def execute(self, action):
                self.calls += 1

            def observe(self):
                stamp = stamps() if callable(stamps) else stamps[0]
                return Observation(
                    objects={"block": "box"}, metadata={"observed_at_monotonic": stamp}
                )

        robot = Robot()

        def sleep(delay):
            test.t += delay

        result = AgentRuntime(
            Planner(),
            MockPolicy(),
            robot,
            confirmation_timeout_s=timeout,
            max_observation_age_s=age,
            clock=lambda: self.t,
            sleep=sleep,
        ).run("verify")
        self.assertEqual(robot.calls, 1)
        return result

    def test_replayed_success_before_action_is_not_confirmed(self):
        self.assertFalse(self.run_case([9.999]).success)

    def test_old_then_new_observation_can_confirm_without_reexecution(self):
        self.assertTrue(self.run_case(lambda: 9.0 if self.t < 10.01 else self.t).success)

    def test_invalid_and_future_timestamps_fail_closed(self):
        for stamp in (None, True, float("nan"), float("inf"), 11.0):
            result = self.run_case([stamp])
            self.assertFalse(result.success)
            self.assertIn("invalid_observation_timestamp", result.message)

    def test_immediate_mode_accepts_current_stamp(self):
        self.assertTrue(self.run_case(lambda: self.t, timeout=0).success)

    def test_invalid_age(self):
        for age in (0, -1, True, float("nan"), 6):
            with self.assertRaises(ValueError):
                AgentRuntime(None, None, None, max_observation_age_s=age)


if __name__ == "__main__":
    unittest.main()
