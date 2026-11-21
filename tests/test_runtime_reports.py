import json
import subprocess
import sys
import unittest

from embodied_agent.mock import MockPolicy, MockRobot
from embodied_agent.models import Action, Observation
from embodied_agent.planner import RuleBasedPlanner
from embodied_agent.runtime import AgentRuntime


class ReportTests(unittest.TestCase):
    def test_reports_are_per_run_and_legacy_result_fields_work(self):
        runtime = AgentRuntime(RuleBasedPlanner(), MockPolicy(), MockRobot())
        result = runtime.run("red block box")
        self.assertEqual(len(result.confirmation_reports), 4)
        self.assertTrue(
            all(r["status"] == "confirmed" and r["reads"] == 1 for r in result.confirmation_reports)
        )
        result.confirmation_reports.clear()
        runtime.robot = MockRobot()
        self.assertEqual(len(runtime.run("red block box").confirmation_reports), 4)

    def test_duplicate_and_negative_counters(self):
        values = iter([(0, True), (0, True), (0.02, False), (0.03, True)])
        t = [0.0]

        class Robot:
            def observe(self):
                stamp, ok = next(values)
                t[0] = stamp
                return Observation(
                    objects={"b": "box" if ok else "table"},
                    metadata={"observed_at_monotonic": stamp},
                )

        runtime = AgentRuntime(
            None,
            None,
            Robot(),
            confirmation_timeout_s=0.1,
            max_observation_age_s=0.1,
            confirmation_samples=3,
            max_confirmation_reads=4,
            clock=lambda: t[0],
            sleep=lambda _: None,
        )
        report = {}
        status = runtime._await_confirmation(
            Action("verify", {"object": "b", "destination": "box"}), lambda: False, report=report
        )
        self.assertEqual(status, "read_limit")
        self.assertEqual(report["duplicate_reads"], 1)
        self.assertEqual(report["negative_reads"], 1)
        self.assertEqual(report["streak_resets"], 1)
        self.assertEqual(report["final_streak"], 1)
        self.assertEqual(report["reads"], 4)

    def test_cli_json_is_machine_readable(self):
        child = subprocess.run(
            [sys.executable, "-m", "embodied_agent", "red block box", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(child.returncode, 0)
        result = json.loads(child.stdout)
        self.assertTrue(result["success"])
        self.assertEqual(len(result["confirmation_reports"]), 4)


if __name__ == "__main__":
    unittest.main()
