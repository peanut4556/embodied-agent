import unittest

from embodied_agent.models import Action, Observation, Plan, PlanStep
from embodied_agent.runtime import AgentRuntime


class TaskBudgetTests(unittest.TestCase):
    def build(self, *, planning=0, selecting=0, executing=0):
        self.t = 10.0
        test = self

        class Planner:
            def create_plan(self, instruction, observation):
                test.t += planning
                return Plan(
                    "verify", [PlanStep("verify", {"object": "b", "destination": "box"})] * 2
                )

        class Policy:
            def select_action(self, step, observation):
                test.t += selecting
                return Action(step.action, step.arguments)

        class Robot:
            calls = 0

            def observe(self):
                return Observation(objects={"b": "box"})

            def execute(self, action):
                self.calls += 1
                test.t += executing

        self.robot = Robot()
        return AgentRuntime(Planner(), Policy(), self.robot, clock=lambda: test.t)

    def test_budget_spans_steps_and_blocks_next_dispatch(self):
        result = self.build(selecting=0.06).run("verify", task_timeout_s=0.1)
        self.assertFalse(result.success)
        self.assertEqual(result.completed_steps, 1)
        self.assertEqual(self.robot.calls, 1)
        self.assertAlmostEqual(result.confirmation_reports[0]["task_remaining_s"], 0.04)

    def test_planning_consumes_budget(self):
        result = self.build(planning=0.2).run("verify", task_timeout_s=0.1)
        self.assertFalse(result.success)
        self.assertEqual(self.robot.calls, 0)

    def test_late_execute_cannot_confirm_success(self):
        result = self.build(executing=0.2).run("verify", task_timeout_s=0.1)
        self.assertFalse(result.success)
        self.assertEqual(result.confirmation_reports[0]["status"], "task_timeout")
        self.assertEqual(result.completed_steps, 0)
        self.assertEqual(self.robot.calls, 1)

    def test_invalid_budget_rejected(self):
        runtime = self.build()
        for value in (True, 0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                runtime.run("verify", task_timeout_s=value)
        self.assertEqual(self.robot.calls, 0)

    def test_no_budget_retains_existing_behavior(self):
        result = self.build(selecting=1).run("verify")
        self.assertTrue(result.success)
        self.assertEqual(self.robot.calls, 2)

    def test_admission_rejects_before_deadline_and_before_execute(self):
        result = self.build(selecting=0.06).run(
            "verify", task_timeout_s=0.1, minimum_step_budget_s=0.05
        )
        self.assertFalse(result.success)
        self.assertEqual(self.robot.calls, 0)
        report = result.confirmation_reports[0]
        self.assertEqual(report["status"], "insufficient_step_budget")
        self.assertFalse(report["dispatched"])
        self.assertGreater(report["task_remaining_s"], 0)

    def test_admission_is_checked_again_for_later_steps(self):
        result = self.build(selecting=0.03).run(
            "verify", task_timeout_s=0.1, minimum_step_budget_s=0.05
        )
        self.assertEqual(self.robot.calls, 1)
        self.assertEqual(result.completed_steps, 1)
        self.assertEqual(result.confirmation_reports[-1]["status"], "insufficient_step_budget")

    def test_admission_validation_and_sufficient_budget(self):
        runtime = self.build()
        for value in (True, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                runtime.run("verify", task_timeout_s=1, minimum_step_budget_s=value)
        with self.assertRaises(ValueError):
            runtime.run("verify", minimum_step_budget_s=0.1)
        self.assertEqual(self.robot.calls, 0)
        self.assertTrue(runtime.run("verify", task_timeout_s=1, minimum_step_budget_s=0.1).success)
