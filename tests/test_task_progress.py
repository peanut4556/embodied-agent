"""A command counts as complete only after acknowledgement and before final scoring."""

import unittest
from unittest.mock import patch

from embodied_agent.models import Observation, Plan, PlanStep
from embodied_agent.planner import RuleBasedPlanner
from embodied_agent.sim_app import SimulationApp


class TaskProgressTests(unittest.TestCase):
    def run_plan(self, mode, *, fail_stop=False, cancel_stop=False, goal=True):
        app = SimulationApp()
        app.task.update(completed=0)
        base = RuleBasedPlanner().create_plan("把红色积木放进盒子", Observation())
        plan = Plan(base.goal, base.steps + [PlanStep("stop", {"reason": "task complete"})])
        commands = []
        state = {
            "active": "",
            "feedback_available": True,
            "objects": {},
            "gripper_holding": "",
            "inside_box": goal,
            "holding": False,
        }

        def bridge(payload=None):
            if payload is None:
                return state
            commands.append(payload["name"])
            if payload["name"] == "stop":
                if fail_stop:
                    raise RuntimeError("stop acknowledgement failed")
                if cancel_stop:
                    app.cancel.set()
            return {"id": "acknowledged", "state": {"execution": {"retries": 0}}}

        with (
            patch("embodied_agent.sim_app.bridge", side_effect=bridge),
            patch("embodied_agent.sim_app.RAISubprocessPlanner.create_plan", return_value=plan),
        ):
            app.run("task", "rai", mode)
        return app.task, commands

    def test_terminal_stop_counts_in_both_execution_modes(self):
        for mode, count in [("scripted", 5), ("feedback", 4)]:
            with self.subTest(mode=mode):
                task, commands = self.run_plan(mode)
                self.assertEqual(task["phase"], "success")
                self.assertEqual(task["completed"], count)
                self.assertEqual(len(task["plan"]), count)
                self.assertEqual(commands[-1], "stop")
                self.assertTrue(any("ROS ← 完成 stop" in e["text"] for e in task["events"]))

    def test_failed_stop_is_not_counted(self):
        task, _ = self.run_plan("feedback", fail_stop=True)
        self.assertEqual(task["phase"], "failed")
        self.assertEqual(task["completed"], 3)
        self.assertEqual(task["current"], 3)
        self.assertFalse(any("ROS ← 完成 stop" in e["text"] for e in task["events"]))

    def test_user_cancellation_is_not_planner_completion(self):
        task, _ = self.run_plan("feedback", cancel_stop=True)
        self.assertEqual(task["phase"], "stopped")
        self.assertEqual(task["completed"], 3)
        self.assertFalse(any("已执行规划器的停止指令" in e["text"] for e in task["events"]))

    def test_full_step_count_does_not_override_failed_goal(self):
        task, _ = self.run_plan("feedback", goal=False)
        self.assertEqual(task["completed"], 4)
        self.assertEqual(task["phase"], "failed")
        self.assertIn("未完成放置任务", task["failure"]["detail"])


if __name__ == "__main__":
    unittest.main()
