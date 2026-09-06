from __future__ import annotations

from dataclasses import replace

from .models import Action, Observation, PlanStep


class MockPolicy:
    def select_action(self, step: PlanStep, observation: Observation) -> Action:
        del observation
        return Action(step.action, dict(step.arguments))


class MockRobot:
    def __init__(self) -> None:
        self._observation = Observation(objects={"red_block": "table", "box": "table"})

    def observe(self) -> Observation:
        return self._observation

    def execute(self, action: Action) -> None:
        objects = dict(self._observation.objects)
        holding = self._observation.gripper_holding
        obj = str(action.parameters.get("object", ""))

        if action.name == "pick":
            if objects.get(obj) != "table":
                raise RuntimeError(f"cannot pick {obj}: object is not on the table")
            holding = obj
            objects[obj] = "gripper"
        elif action.name == "place":
            destination = str(action.parameters.get("destination", ""))
            if holding != obj:
                raise RuntimeError(f"cannot place {obj}: gripper is not holding it")
            holding = ""
            objects[obj] = destination

        self._observation = replace(
            self._observation,
            objects=objects,
            gripper_holding=holding,
            metadata={"last_action": action.name},
        )
