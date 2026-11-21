import unittest
from unittest.mock import patch

from embodied_agent import cli
from embodied_agent.mock import MockPolicy, TimestampedMockRobot
from embodied_agent.models import Observation, Plan, PlanStep, TaskResult
from embodied_agent.runtime import AgentRuntime


class Planner:
    def create_plan(self, *args):
        return Plan("verify", [PlanStep("verify", {"object": "block", "destination": "box"})])


class MultiTests(unittest.TestCase):
    def run_sequence(self, values):
        t = [10.0]

        class Robot:
            calls = 0
            index = 0

            def execute(self, action):
                self.calls += 1

            def observe(self):
                if not self.calls:
                    return Observation()
                positive, stamp = values[min(self.index, len(values) - 1)]
                self.index += 1
                t[0] += 0.001
                return Observation(
                    objects={"block": "box" if positive else "table"},
                    metadata={"observed_at_monotonic": stamp},
                )

        robot = Robot()
        result = AgentRuntime(
            Planner(),
            MockPolicy(),
            robot,
            confirmation_timeout_s=0.1,
            max_observation_age_s=0.1,
            confirmation_samples=2,
            max_confirmation_reads=6,
            clock=lambda: t[0],
            sleep=lambda delay: t.__setitem__(0, t[0] + delay),
        ).run("verify")
        self.assertEqual(robot.calls, 1)
        return result, robot.index

    def test_distinct_samples_confirm(self):
        result, reads = self.run_sequence([(True, 10.0), (True, 10.01)])
        self.assertTrue(result.success)
        self.assertEqual(reads, 2)

    def test_replayed_frame_never_accumulates(self):
        self.assertFalse(self.run_sequence([(True, 10.0)])[0].success)

    def test_negative_resets_streak(self):
        result, reads = self.run_sequence(
            [(True, 10), (False, 10.01), (True, 10.02), (True, 10.03)]
        )
        self.assertTrue(result.success)
        self.assertEqual(reads, 4)

    def test_timestamp_regression_rejected(self):
        result, _ = self.run_sequence([(True, 10.001), (True, 10)])
        self.assertIn("regressed", result.message)

    def test_invalid_combination(self):
        for kwargs in (
            {"confirmation_samples": True},
            {"confirmation_samples": 11},
            {"confirmation_samples": 2},
            {"confirmation_samples": 2, "max_observation_age_s": 0.1},
        ):
            with self.assertRaises(ValueError):
                AgentRuntime(None, None, None, **kwargs)

    def test_cli_wires_freshness_and_sample_count(self):
        with (
            patch(
                "sys.argv",
                [
                    "agent",
                    "--confirmation-timeout",
                    ".5",
                    "--max-observation-age",
                    ".1",
                    "--confirmation-samples",
                    "2",
                ],
            ),
            patch("embodied_agent.cli.AgentRuntime") as runtime,
            patch("builtins.print"),
        ):
            runtime.return_value.run.return_value = TaskResult(True, "ok", 4)
            cli.main()
        self.assertIsInstance(runtime.call_args.args[2], TimestampedMockRobot)
        self.assertEqual(runtime.call_args.kwargs["max_observation_age_s"], 0.1)
        self.assertEqual(runtime.call_args.kwargs["confirmation_samples"], 2)


if __name__ == "__main__":
    unittest.main()
