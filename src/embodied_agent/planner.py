from __future__ import annotations

from .models import Observation, Plan, PlanStep


class RuleBasedPlanner:
    """Deterministic bootstrap planner; replace with a RAI-backed planner later."""

    def create_plan(self, instruction: str, observation: Observation) -> Plan:
        del observation
        if not instruction.strip():
            raise ValueError("instruction must not be empty")

        target = "red_block" if "红" in instruction or "red" in instruction.lower() else "object"
        destination = "box" if "盒" in instruction or "box" in instruction.lower() else "target"
        return Plan(
            goal=instruction,
            steps=[
                PlanStep("locate", {"object": target}, f"{target} is localized"),
                PlanStep("pick", {"object": target}, f"gripper holds {target}"),
                PlanStep(
                    "place",
                    {"object": target, "destination": destination},
                    f"{target} is in {destination}",
                ),
                PlanStep(
                    "verify",
                    {"object": target, "destination": destination},
                    "goal is visually verified",
                ),
            ],
        )
