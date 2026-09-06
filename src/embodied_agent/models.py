from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlanStep:
    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success_condition: str = ""


@dataclass(frozen=True)
class Plan:
    goal: str
    steps: list[PlanStep]


@dataclass(frozen=True)
class Observation:
    objects: dict[str, str] = field(default_factory=dict)
    gripper_holding: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskResult:
    success: bool
    message: str
    completed_steps: int
