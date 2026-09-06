from __future__ import annotations

from typing import Protocol

from .models import Action, Observation, Plan, PlanStep


class Planner(Protocol):
    def create_plan(self, instruction: str, observation: Observation) -> Plan:
        """Convert a user instruction into bounded, structured steps."""


class Policy(Protocol):
    def select_action(self, step: PlanStep, observation: Observation) -> Action:
        """Convert one plan step and the current observation into an action."""


class Robot(Protocol):
    def observe(self) -> Observation:
        """Return the latest normalized robot/world observation."""

    def execute(self, action: Action) -> None:
        """Execute one previously validated action."""
