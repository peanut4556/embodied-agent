from __future__ import annotations

import copy
import logging

from .interfaces import Planner, Policy, Robot
from .models import Action, Observation, TaskResult
from .safety import SafetyGate

LOGGER = logging.getLogger(__name__)


class AgentRuntime:
    def __init__(
        self,
        planner: Planner,
        policy: Policy,
        robot: Robot,
        safety: SafetyGate | None = None,
    ) -> None:
        self.planner = planner
        self.policy = policy
        self.robot = robot
        self.safety = safety or SafetyGate()

    def run(self, instruction: str) -> TaskResult:
        observation = self.robot.observe()
        plan = self.planner.create_plan(instruction, observation)
        self.safety.validate_plan(plan)
        LOGGER.info("goal=%s steps=%d", plan.goal, len(plan.steps))

        completed = 0
        for index, step in enumerate(plan.steps, start=1):
            observation = self.robot.observe()
            action = self.policy.select_action(copy.deepcopy(step), observation)
            self.safety.validate_action(action)
            if action.name != step.action or any(
                action.parameters.get(key) != value for key, value in step.arguments.items()
            ):
                return TaskResult(
                    False, f"step {index}: policy action differs from validated plan", completed
                )
            LOGGER.info("step=%d action=%s parameters=%s", index, action.name, action.parameters)
            self.robot.execute(action)
            if action.name == "stop":
                return TaskResult(False, "task stopped", completed)
            after = self.robot.observe()
            if not self._confirmed(action, after):
                return TaskResult(
                    False, f"step {index}: {action.name} postcondition not confirmed", completed
                )
            completed += 1

        return TaskResult(True, "task completed", completed)

    @staticmethod
    def _confirmed(action: Action, observation: Observation) -> bool:
        """Symbolic adapter observations only; not independent physical verification."""
        target = action.parameters.get("object")
        if action.name == "locate":
            return target in observation.objects
        if action.name == "pick":
            return (
                observation.gripper_holding == target
                and observation.objects.get(target) == "gripper"
            )
        if action.name in {"place", "verify"}:
            destination = action.parameters.get("destination")
            return (
                isinstance(destination, str)
                and bool(destination)
                and observation.objects.get(target) == destination
                and observation.gripper_holding == ""
            )
        return False
