import unittest
from unittest.mock import patch

from embodied_agent.execution_plan import execution_steps
from embodied_agent.models import Observation, Plan, PlanStep
from embodied_agent.planner import RuleBasedPlanner
from embodied_agent.sim_app import SimulationApp


class ExecutionPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = RuleBasedPlanner().create_plan("把红色积木放进盒子", Observation())

    def test_compiles_only_pick_place_and_preserves_original(self):
        steps = execution_steps(self.plan, "feedback")
        self.assertEqual([s.action for s in steps], ["locate", "learned_pick_place", "verify"])
        self.assertEqual([s.action for s in self.plan.steps], ["locate", "pick", "place", "verify"])
        self.assertEqual(execution_steps(self.plan, "scripted"), self.plan.steps)

    def test_accepts_no_locate_and_verify_without_destination(self):
        plan = Plan("task", self.plan.steps[1:3] + [PlanStep("verify", {"object": "red_block"})])
        self.assertEqual(len(execution_steps(plan, "feedback")), 2)

    def test_rejects_unsupported_semantics_before_execution(self):
        variants = [
            [PlanStep("stop")] + self.plan.steps,
            self.plan.steps[:2] + [PlanStep("stop")] + self.plan.steps[2:],
            self.plan.steps[:-1],
            [PlanStep("learned_pick_place", {"object": "red_block", "destination": "box"})],
            [PlanStep("locate", {"object": "box"})] + self.plan.steps[1:],
            self.plan.steps[:2]
            + [PlanStep("place", {"object": "red_block", "destination": "shelf"})]
            + self.plan.steps[3:],
        ]
        for steps in variants:
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                execution_steps(Plan("task", steps), "feedback")

    def test_qwen_null_destination_and_final_stop_are_preserved(self):
        stop = PlanStep("stop", {"reason": "task complete"})
        plan = Plan(
            "task",
            self.plan.steps[:3]
            + [
                PlanStep("verify", {"object": "red_block", "destination": None}),
                stop,
            ],
        )
        steps = execution_steps(plan, "feedback")
        self.assertEqual(
            [s.action for s in steps], ["locate", "learned_pick_place", "verify", "stop"]
        )
        self.assertEqual(steps[-1], stop)

    def test_missing_model_fails_without_dispatching_motion(self):
        app = SimulationApp()
        with patch("embodied_agent.sim_app.bridge", return_value={"active": ""}) as bridge:
            app.run("task", "rule", "feedback")
        self.assertEqual(app.task["phase"], "failed")
        self.assertEqual([c.args[0]["name"] for c in bridge.call_args_list if c.args], ["stop"])

    def test_unsupported_plan_never_falls_back_to_script(self):
        app = SimulationApp()
        state = {"active": "", "feedback_available": True, "objects": {}, "gripper_holding": ""}
        with (
            patch("embodied_agent.sim_app.bridge", return_value=state) as bridge,
            patch(
                "embodied_agent.sim_app.RuleBasedPlanner.create_plan",
                return_value=Plan("stop", [PlanStep("stop")]),
            ),
        ):
            app.run("task", "rule", "feedback")
        self.assertEqual(app.task["phase"], "failed")
        self.assertEqual([c.args[0]["name"] for c in bridge.call_args_list if c.args], ["stop"])
