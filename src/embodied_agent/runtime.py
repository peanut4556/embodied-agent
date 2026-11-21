from __future__ import annotations

import copy
import logging
import math
import time

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
        *,
        confirmation_timeout_s=0,
        confirmation_poll_s=0.01,
        max_confirmation_reads=128,
        max_observation_age_s=None,
        confirmation_samples=1,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        for value, low, high in ((confirmation_timeout_s, 0, 5), (confirmation_poll_s, 0.001, 1)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError("invalid confirmation timing")
        if type(max_confirmation_reads) is not int or not 1 <= max_confirmation_reads <= 1000:
            raise ValueError("confirmation read limit must be 1..1000")
        if not callable(clock) or not callable(sleep):
            raise TypeError("clock and sleep must be callable")
        if max_observation_age_s is not None and (
            isinstance(max_observation_age_s, bool)
            or not isinstance(max_observation_age_s, (int, float))
            or not math.isfinite(max_observation_age_s)
            or not 0 < max_observation_age_s <= 5
        ):
            raise ValueError("observation age must be None or finite seconds in (0, 5]")
        if type(confirmation_samples) is not int or not 1 <= confirmation_samples <= 10:
            raise ValueError("confirmation samples must be 1..10")
        if confirmation_samples > 1 and (
            max_observation_age_s is None
            or confirmation_timeout_s <= 0
            or max_confirmation_reads < confirmation_samples
        ):
            raise ValueError(
                "multiple samples require freshness, positive timeout and enough reads"
            )
        self.confirmation_samples = confirmation_samples
        self.max_observation_age_s = max_observation_age_s
        self.confirmation_timeout_s = confirmation_timeout_s
        self.confirmation_poll_s = confirmation_poll_s
        self.max_confirmation_reads = max_confirmation_reads
        self.clock, self.sleep = clock, sleep
        self.planner = planner
        self.policy = policy
        self.robot = robot
        self.safety = safety or SafetyGate()

    def run(self, instruction: str, *, stop_requested=None) -> TaskResult:
        if stop_requested is not None and not callable(stop_requested):
            raise ValueError("stop_requested must be callable")
        stopped = stop_requested if stop_requested is not None else lambda: False
        if stopped():
            return TaskResult(False, "task cancelled", 0)
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
            if stopped():
                return TaskResult(False, "task cancelled", completed)
            self.robot.execute(action)
            if action.name == "stop":
                return TaskResult(False, "task stopped", completed)
            confirmation = self._await_confirmation(action, stopped)
            if confirmation != "confirmed":
                return TaskResult(
                    False,
                    f"step {index}: {action.name} postcondition not confirmed ({confirmation})",
                    completed,
                )
            completed += 1

        return TaskResult(True, "task completed", completed)

    def _await_confirmation(self, action, stopped):
        previous = None

        def now():
            nonlocal previous
            value = self.clock()
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (previous is not None and value < previous)
            ):
                raise ValueError("confirmation clock invalid")
            previous = value
            return value

        started = now()
        deadline = started + self.confirmation_timeout_s
        streak, last_stamp = 0, None
        for attempt in range(self.max_confirmation_reads):
            if stopped():
                return "cancelled"
            if attempt and now() >= deadline:
                return "timeout"
            after = self.robot.observe()
            current = now()
            if stopped():
                return "cancelled"
            if self.confirmation_timeout_s and current > deadline:
                return "timeout"
            fresh = True
            if self.max_observation_age_s is not None:
                stamp = after.metadata.get("observed_at_monotonic")
                if (
                    isinstance(stamp, bool)
                    or not isinstance(stamp, (int, float))
                    or not math.isfinite(stamp)
                    or stamp > current
                ):
                    return "invalid_observation_timestamp"
                fresh = stamp >= started and current - stamp <= self.max_observation_age_s
            positive = fresh and self._confirmed(action, after)
            if self.confirmation_samples == 1 and positive:
                return "confirmed"
            if self.confirmation_samples > 1:
                if last_stamp is not None and stamp < last_stamp:
                    return "observation_timestamp_regressed"
                if not positive:
                    streak = 0
                elif last_stamp is None or stamp > last_stamp:
                    streak += 1
                last_stamp = stamp
                if streak >= self.confirmation_samples:
                    return "confirmed"
            if not self.confirmation_timeout_s or current >= deadline:
                return "timeout"
            if attempt + 1 < self.max_confirmation_reads:
                self.sleep(min(self.confirmation_poll_s, deadline - current))
        return "read_limit"

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
