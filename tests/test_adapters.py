import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from embodied_agent.adapters.lerobot_policy import LeRobotPolicyAdapter
from embodied_agent.adapters.rai_planner import (
    RAIPlannerAdapter,
    RAISubprocessPlanner,
    decode_plan,
)
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

    def test_rai_subprocess_planner_exchanges_json(self):
        with TemporaryDirectory() as temp_dir:
            worker = Path(temp_dir) / "worker.py"
            worker.write_text("# test worker", encoding="utf-8")
            planner = RAISubprocessPlanner(
                python_executable="python3.12",
                worker_script=worker,
                config_path="config.toml",
            )
            completed = type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": json.dumps(
                        {
                            "goal": "检查积木",
                            "steps": [
                                {
                                    "action": "verify",
                                    "arguments": {"object": "red_block"},
                                    "success_condition": "积木位置已确认",
                                }
                            ],
                        }
                    ),
                    "stderr": "",
                },
            )()
            with patch("subprocess.run", return_value=completed) as run:
                plan = planner.create_plan("检查积木", Observation(objects={"red_block": "box"}))

        request = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(request["instruction"], "检查积木")
        self.assertEqual(request["observation"]["objects"]["red_block"], "box")
        self.assertEqual(plan.steps[0].success_condition, "积木位置已确认")

    def test_decode_plan_rejects_invalid_arguments(self):
        with self.assertRaisesRegex(TypeError, "arguments"):
            decode_plan(
                {
                    "goal": "bad plan",
                    "steps": [{"action": "pick", "arguments": "red_block"}],
                }
            )

    def test_ros2_adapter_serializes_normalized_action(self):
        messages = []
        observation = Observation(objects={"red_block": "table"})
        adapter = ROS2RobotAdapter(lambda: observation, messages.append)

        adapter.execute(Action("pick", {"object": "red_block"}))

        self.assertEqual(adapter.observe(), observation)
        self.assertEqual(json.loads(messages[0])["name"], "pick")


if __name__ == "__main__":
    unittest.main()
