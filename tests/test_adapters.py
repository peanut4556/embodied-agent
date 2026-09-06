import json
import unittest

from embodied_agent.adapters.lerobot_policy import LeRobotPolicyAdapter
from embodied_agent.adapters.rai_planner import RAIPlannerAdapter
from embodied_agent.adapters.ros2_robot import ROS2RobotAdapter
from embodied_agent.models import Action, Observation, Plan, PlanStep


class FakeLeRobotPolicy:
    def __init__(self):
        self.last_input = None

    def select_action(self, model_input):
        self.last_input = model_input
        return [0.1, 0.2]


class AdapterTests(unittest.TestCase):
    def test_lerobot_adapter_keeps_robot_specific_codec_at_boundary(self):
        policy = FakeLeRobotPolicy()
        adapter = LeRobotPolicyAdapter(
            policy,
            encode_observation=lambda step, observation: {
                "task": step.action,
                "objects": observation.objects,
            },
            decode_action=lambda raw: Action("joint_delta", {"values": raw}),
        )

        action = adapter.select_action(
            PlanStep("pick", {"object": "red_block"}),
            Observation(objects={"red_block": "table"}),
        )

        self.assertEqual(policy.last_input["task"], "pick")
        self.assertEqual(action.parameters["values"], [0.1, 0.2])

    def test_rai_adapter_decodes_structured_plan(self):
        adapter = RAIPlannerAdapter(
            invoke_agent=lambda instruction, observation: {
                "goal": instruction,
                "steps": [{"action": "verify", "arguments": observation.objects}],
            },
            decode_plan=lambda payload: Plan(
                payload["goal"],
                [PlanStep(item["action"], item["arguments"]) for item in payload["steps"]],
            ),
        )

        plan = adapter.create_plan("检查积木", Observation(objects={"red_block": "box"}))

        self.assertEqual(plan.goal, "检查积木")
        self.assertEqual(plan.steps[0].action, "verify")

    def test_ros2_adapter_serializes_normalized_action(self):
        messages = []
        observation = Observation(objects={"red_block": "table"})
        adapter = ROS2RobotAdapter(lambda: observation, messages.append)

        adapter.execute(Action("pick", {"object": "red_block"}))

        self.assertEqual(adapter.observe(), observation)
        self.assertEqual(json.loads(messages[0])["name"], "pick")


if __name__ == "__main__":
    unittest.main()
