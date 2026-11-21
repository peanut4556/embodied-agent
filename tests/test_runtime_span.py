import unittest
from unittest.mock import patch

from embodied_agent import cli
from embodied_agent.models import Action, Observation, TaskResult
from embodied_agent.runtime import AgentRuntime


class SpanTests(unittest.TestCase):
    def confirm(self, samples, span=0.03, gap=0.02):
        t = [0.0]

        class Robot:
            reads = 0

            def observe(self):
                stamp, positive = samples[min(self.reads, len(samples) - 1)]
                self.reads += 1
                t[0] = max(t[0], stamp)
                return Observation(
                    objects={"block": "box" if positive else "table"},
                    metadata={"observed_at_monotonic": stamp},
                )

        robot = Robot()
        runtime = AgentRuntime(
            None,
            None,
            robot,
            confirmation_timeout_s=0.3,
            max_observation_age_s=0.3,
            confirmation_samples=2,
            confirmation_span_s=span,
            max_confirmation_gap_s=gap,
            max_confirmation_reads=len(samples),
            clock=lambda: t[0],
            sleep=lambda _: None,
        )
        return runtime._await_confirmation(
            Action("verify", {"object": "block", "destination": "box"}), lambda: False
        ), robot.reads

    def test_dense_burst_cannot_meet_span(self):
        self.assertEqual(self.confirm([(0, True), (0.001, True), (0.002, True)])[0], "read_limit")

    def test_spanning_samples_confirm(self):
        self.assertEqual(self.confirm([(0, True), (0.015, True), (0.031, True)]), ("confirmed", 3))

    def test_long_gap_restarts_span(self):
        self.assertEqual(
            self.confirm([(0, True), (0.015, True), (0.1, True), (0.115, True)])[0], "read_limit"
        )
        self.assertEqual(
            self.confirm([(0, True), (0.015, True), (0.1, True), (0.115, True), (0.132, True)])[0],
            "confirmed",
        )

    def test_negative_restarts_span_and_duplicate_does_not_extend(self):
        self.assertEqual(
            self.confirm([(0, True), (0.015, False), (0.031, True), (0.045, True)])[0], "read_limit"
        )
        self.assertEqual(self.confirm([(0, True), (0.015, True), (0.015, True)])[0], "read_limit")

    def test_invalid_config(self):
        base = {
            "confirmation_timeout_s": 0.5,
            "max_observation_age_s": 0.1,
            "confirmation_samples": 2,
        }
        for extra in (
            {"confirmation_span_s": 0.1},
            {"confirmation_span_s": True},
            {"confirmation_span_s": 0.6},
            {"max_confirmation_gap_s": 0},
        ):
            with self.assertRaises(ValueError):
                AgentRuntime(None, None, None, **base, **extra)

    def test_cli_transmits_temporal_options(self):
        with (
            patch(
                "sys.argv", ["agent", "--confirmation-span", ".05", "--max-confirmation-gap", ".03"]
            ),
            patch("embodied_agent.cli.AgentRuntime") as runtime,
            patch("builtins.print"),
        ):
            runtime.return_value.run.return_value = TaskResult(True, "ok", 4)
            cli.main()
        self.assertEqual(runtime.call_args.kwargs["confirmation_span_s"], 0.05)
        self.assertEqual(runtime.call_args.kwargs["max_confirmation_gap_s"], 0.03)


if __name__ == "__main__":
    unittest.main()
