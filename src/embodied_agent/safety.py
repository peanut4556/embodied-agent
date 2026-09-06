from __future__ import annotations

from dataclasses import dataclass

from .models import Action, Plan


@dataclass(frozen=True)
class SafetyGate:
    allowed_actions: frozenset[str] = frozenset({"locate", "pick", "place", "verify", "stop"})
    max_plan_steps: int = 8

    def validate_plan(self, plan: Plan) -> None:
        if not plan.steps:
            raise ValueError("refusing an empty plan")
        if len(plan.steps) > self.max_plan_steps:
            raise ValueError(f"plan has {len(plan.steps)} steps; limit is {self.max_plan_steps}")

    def validate_action(self, action: Action) -> None:
        if action.name not in self.allowed_actions:
            raise ValueError(f"action is not allowed: {action.name}")
