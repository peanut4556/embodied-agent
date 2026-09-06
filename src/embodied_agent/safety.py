from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .models import Action, Plan


@dataclass(frozen=True)
class SafetyGate:
    allowed_actions: frozenset[str] = frozenset({"locate", "pick", "place", "verify", "stop"})
    max_plan_steps: int = 8

    required_parameters: ClassVar[dict[str, frozenset[str]]] = {
        "locate": frozenset({"object"}),
        "pick": frozenset({"object"}),
        "place": frozenset({"object", "destination"}),
        "verify": frozenset({"object"}),
        "stop": frozenset(),
    }

    def validate_plan(self, plan: Plan) -> None:
        if not plan.steps:
            raise ValueError("refusing an empty plan")
        if len(plan.steps) > self.max_plan_steps:
            raise ValueError(f"plan has {len(plan.steps)} steps; limit is {self.max_plan_steps}")

        last_picked_object = ""
        for step in plan.steps:
            action = Action(step.action, step.arguments)
            self.validate_action(action)
            if action.name == "pick":
                last_picked_object = str(action.parameters["object"])
            elif action.name == "place" and last_picked_object:
                placed_object = str(action.parameters["object"])
                if placed_object != last_picked_object:
                    raise ValueError(
                        "place object must match the most recently picked object: "
                        f"picked={last_picked_object}, place={placed_object}"
                    )

    def validate_action(self, action: Action) -> None:
        if action.name not in self.allowed_actions:
            raise ValueError(f"action is not allowed: {action.name}")
        required = self.required_parameters[action.name]
        missing = sorted(
            name for name in required if not str(action.parameters.get(name, "")).strip()
        )
        if missing:
            raise ValueError(
                f"action {action.name} is missing required parameters: {', '.join(missing)}"
            )
