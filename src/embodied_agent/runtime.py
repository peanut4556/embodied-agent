from __future__ import annotations

import logging

from .interfaces import Planner, Policy, Robot
from .models import TaskResult
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
            action = self.policy.select_action(step, observation)
            self.safety.validate_action(action)
            LOGGER.info("step=%d action=%s parameters=%s", index, action.name, action.parameters)
            self.robot.execute(action)
            completed += 1

        return TaskResult(True, "task completed", completed)
