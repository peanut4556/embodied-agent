from __future__ import annotations

import copy
import logging
import math
import time

from .interfaces import Planner, Policy, Robot
from .models import Action, Observation, TaskResult
from .process_witness import (
    ProcessWitness,
    WitnessCancelled,
    WitnessDeadline,
    WitnessFailure,
    WitnessTimeout,
)
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
        confirmation_span_s=0,
        max_confirmation_gap_s=None,
        confirmation_witness=None,
        max_witness_skew_s=0.02,
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
        if (
            isinstance(confirmation_span_s, bool)
            or not isinstance(confirmation_span_s, (int, float))
            or not math.isfinite(confirmation_span_s)
            or not 0 <= confirmation_span_s <= confirmation_timeout_s
        ):
            raise ValueError("confirmation span must be finite and within timeout")
        if max_confirmation_gap_s is not None and (
            isinstance(max_confirmation_gap_s, bool)
            or not isinstance(max_confirmation_gap_s, (int, float))
            or not math.isfinite(max_confirmation_gap_s)
            or not 0 < max_confirmation_gap_s <= 5
        ):
            raise ValueError("confirmation gap must be finite seconds in (0, 5]")
        if (confirmation_span_s or max_confirmation_gap_s is not None) and confirmation_samples < 2:
            raise ValueError("temporal confirmation requires multiple samples")
        if confirmation_span_s and max_confirmation_gap_s is None:
            raise ValueError("confirmation span requires an explicit maximum sample gap")
        if confirmation_witness is not None and (
            not callable(confirmation_witness) or max_observation_age_s is None
        ):
            raise ValueError("confirmation witness requires a callable and freshness limit")
        if (
            isinstance(max_witness_skew_s, bool)
            or not isinstance(max_witness_skew_s, (int, float))
            or not math.isfinite(max_witness_skew_s)
            or not 0 <= max_witness_skew_s <= 0.1
        ):
            raise ValueError("witness skew must be finite seconds in [0, .1]")
        self.confirmation_witness = confirmation_witness
        self.max_witness_skew_s = max_witness_skew_s
        self.confirmation_span_s = confirmation_span_s
        self.max_confirmation_gap_s = max_confirmation_gap_s
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

        reports = []
        completed = 0
        for index, step in enumerate(plan.steps, start=1):
            observation = self.robot.observe()
            action = self.policy.select_action(copy.deepcopy(step), observation)
            self.safety.validate_action(action)
            if action.name != step.action or any(
                action.parameters.get(key) != value for key, value in step.arguments.items()
            ):
                return TaskResult(
                    False,
                    f"step {index}: policy action differs from validated plan",
                    completed,
                    reports,
                )
            LOGGER.info("step=%d action=%s parameters=%s", index, action.name, action.parameters)
            if stopped():
                return TaskResult(False, "task cancelled", completed, reports)
            self.robot.execute(action)
            if action.name == "stop":
                return TaskResult(False, "task stopped", completed, reports)
            report = {"step": index, "action": action.name}
            reports.append(report)
            confirmation = self._await_confirmation(action, stopped, report=report)
            if confirmation != "confirmed":
                return TaskResult(
                    False,
                    f"step {index}: {action.name} postcondition not confirmed ({confirmation})",
                    completed,
                    reports,
                )
            completed += 1

        return TaskResult(True, "task completed", completed, reports)

    def _await_confirmation(self, action, stopped, *, report=None):
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

        report = {} if report is None else report
        report.update(
            reads=0,
            stale_reads=0,
            negative_reads=0,
            duplicate_reads=0,
            streak_resets=0,
            gap_resets=0,
            final_streak=0,
            witness_reads=0,
            witness_disagreements=0,
            witness_unaligned=0,
            required_samples=self.confirmation_samples,
            required_span_s=self.confirmation_span_s,
        )
        started = now()

        def finish(status):
            if isinstance(self.confirmation_witness, ProcessWitness):
                report["witness_process"] = self.confirmation_witness.lifecycle_report()
            report.update(status=status, elapsed_s=previous - started)
            return status

        deadline = started + self.confirmation_timeout_s
        streak, last_stamp, first_stamp = 0, None, None
        last_witness_stamp = None
        for attempt in range(self.max_confirmation_reads):
            if stopped():
                return finish("cancelled")
            if attempt and now() >= deadline:
                return finish("timeout")
            after = self.robot.observe()
            report["reads"] += 1
            current = now()
            if stopped():
                return finish("cancelled")
            if self.confirmation_timeout_s and current > deadline:
                return finish("timeout")
            fresh = True
            if self.max_observation_age_s is not None:
                stamp = after.metadata.get("observed_at_monotonic")
                if (
                    isinstance(stamp, bool)
                    or not isinstance(stamp, (int, float))
                    or not math.isfinite(stamp)
                    or stamp > current
                ):
                    return finish("invalid_observation_timestamp")
                fresh = stamp >= started and current - stamp <= self.max_observation_age_s
            report["stale_reads"] += not fresh
            matches = self._confirmed(action, after)
            report["negative_reads"] += not matches
            positive = fresh and matches
            witness_advanced, witness_gap = True, False
            span_start = span_end = stamp if self.max_observation_age_s is not None else current
            if self.confirmation_witness is not None:
                try:
                    if isinstance(self.confirmation_witness, ProcessWitness):
                        budget = max(0, deadline - now()) if self.confirmation_timeout_s else None
                        witness = self.confirmation_witness(
                            copy.deepcopy(action), stopped=stopped, budget_s=budget
                        )
                    else:
                        witness = self.confirmation_witness(copy.deepcopy(action))
                except WitnessCancelled:
                    now()
                    return finish("cancelled")
                except WitnessDeadline:
                    now()
                    return finish("timeout")
                except WitnessTimeout:
                    now()
                    return finish("witness_timeout")
                except WitnessFailure as exc:
                    now()
                    report["witness_failure"] = exc.code
                    return finish("witness_failed")
                report["witness_reads"] += 1
                current = now()
                if stopped():
                    return finish("cancelled")
                if self.confirmation_timeout_s and current > deadline:
                    return finish("timeout")
                if not isinstance(witness, Observation):
                    return finish("invalid_witness_observation")
                witness_stamp = witness.metadata.get("observed_at_monotonic")
                if (
                    isinstance(witness_stamp, bool)
                    or not isinstance(witness_stamp, (int, float))
                    or not math.isfinite(witness_stamp)
                    or witness_stamp > current
                ):
                    return finish("invalid_witness_timestamp")
                if last_witness_stamp is not None and witness_stamp < last_witness_stamp:
                    return finish("witness_timestamp_regressed")
                aligned = (
                    witness_stamp >= started
                    and current - witness_stamp <= self.max_observation_age_s
                    and current - stamp <= self.max_observation_age_s
                    and abs(witness_stamp - stamp) <= self.max_witness_skew_s
                )
                agrees = self._confirmed(action, witness)
                report["witness_disagreements"] += matches != agrees
                report["witness_unaligned"] += not aligned
                positive = positive and aligned and agrees
                witness_advanced = last_witness_stamp is None or witness_stamp > last_witness_stamp
                witness_gap = (
                    last_witness_stamp is not None
                    and self.max_confirmation_gap_s is not None
                    and witness_stamp - last_witness_stamp > self.max_confirmation_gap_s
                )
                last_witness_stamp = witness_stamp
                span_start, span_end = max(stamp, witness_stamp), min(stamp, witness_stamp)
            if self.confirmation_samples == 1 and positive:
                report["final_streak"] = 1
                return finish("confirmed")
            if self.confirmation_samples > 1:
                if last_stamp is not None and stamp < last_stamp:
                    return finish("observation_timestamp_regressed")
                if positive and last_stamp is not None and stamp == last_stamp:
                    report["duplicate_reads"] += 1
                if not positive:
                    report["streak_resets"] += streak > 0
                    streak, first_stamp = 0, None
                elif (last_stamp is None or stamp > last_stamp) and witness_advanced:
                    if (
                        last_stamp is not None
                        and self.max_confirmation_gap_s is not None
                        and (stamp - last_stamp > self.max_confirmation_gap_s or witness_gap)
                    ):
                        report["gap_resets"] += 1
                        streak, first_stamp = 0, None
                    if streak == 0:
                        first_stamp = span_start
                    streak += 1
                last_stamp = stamp
                report["final_streak"] = streak
                report["observed_span_s"] = (
                    max(0, span_end - first_stamp) if first_stamp is not None else 0
                )
                if (
                    streak >= self.confirmation_samples
                    and first_stamp is not None
                    and span_end - first_stamp >= self.confirmation_span_s
                ):
                    return finish("confirmed")
            if not self.confirmation_timeout_s or current >= deadline:
                return finish("timeout")
            if attempt + 1 < self.max_confirmation_reads:
                self.sleep(min(self.confirmation_poll_s, deadline - current))
        return finish("read_limit")

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
