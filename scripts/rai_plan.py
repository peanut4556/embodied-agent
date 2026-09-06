"""RAI worker for generating a bounded robot task plan with a local LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from rai import get_llm_model


class ObjectArguments(BaseModel):
    object: str


class PlaceArguments(BaseModel):
    object: str
    destination: str


class VerifyArguments(BaseModel):
    object: str
    destination: str | None = None


class StopArguments(BaseModel):
    reason: str = ""


class LocateStep(BaseModel):
    action: Literal["locate"]
    arguments: ObjectArguments
    success_condition: str = ""


class PickStep(BaseModel):
    action: Literal["pick"]
    arguments: ObjectArguments
    success_condition: str = ""


class PlaceStep(BaseModel):
    action: Literal["place"]
    arguments: PlaceArguments
    success_condition: str = ""


class VerifyStep(BaseModel):
    action: Literal["verify"]
    arguments: VerifyArguments
    success_condition: str = ""


class StopStep(BaseModel):
    action: Literal["stop"]
    arguments: StopArguments = Field(default_factory=StopArguments)
    success_condition: str = ""


StepPayload = Annotated[
    LocateStep | PickStep | PlaceStep | VerifyStep | StopStep,
    Field(discriminator="action"),
]


class PlanPayload(BaseModel):
    goal: str
    steps: list[StepPayload] = Field(min_length=1, max_length=8)


def semantic_error(plan: PlanPayload) -> str:
    last_picked_object = ""
    for step in plan.steps:
        if isinstance(step, PickStep):
            last_picked_object = step.arguments.object
        elif (
            isinstance(step, PlaceStep)
            and last_picked_object
            and step.arguments.object != last_picked_object
        ):
            return (
                "A place step must move the same object as the preceding pick step; "
                f"picked {last_picked_object!r} but tried to place "
                f"{step.arguments.object!r}."
            )
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = json.load(sys.stdin)
    instruction = request.get("instruction", "")
    observation = request.get("observation", {})
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")

    llm = get_llm_model(
        model_type="complex_model",
        config_path=str(args.config),
        temperature=0,
        reasoning=False,
        num_ctx=8192,
        num_predict=1000,
        validate_model_on_init=True,
    )
    structured_llm = llm.with_structured_output(PlanPayload, method="json_schema")
    messages = [
        SystemMessage(
            content=(
                "You are a conservative robot task planner. Produce a short plan using only "
                "the actions locate, pick, place, verify, and stop. Never invent objects that "
                "are absent from the observation. Prefer stop when the request cannot be "
                "completed safely. Use snake_case object identifiers. A place step must use the "
                "same arguments.object as the preceding pick step. Put the destination name only "
                "in arguments.destination, never in arguments.object, target, or location."
            )
        ),
        HumanMessage(
            content=(
                f"Task instruction: {instruction}\n"
                "Current normalized observation:\n"
                f"{json.dumps(observation, ensure_ascii=False, sort_keys=True)}"
            )
        ),
    ]
    plan: PlanPayload | None = None
    for _attempt in range(3):
        candidate = structured_llm.invoke(messages)
        plan = (
            candidate
            if isinstance(candidate, PlanPayload)
            else PlanPayload.model_validate(candidate)
        )
        error = semantic_error(plan)
        if not error:
            break
        messages.append(
            HumanMessage(
                content=(
                    f"The previous plan was rejected: {error}\n"
                    f"Rejected plan: {plan.model_dump_json()}\n"
                    "Return a corrected complete plan."
                )
            )
        )
    else:
        raise ValueError(f"model failed semantic plan validation: {error}")

    assert plan is not None
    sys.stdout.write(plan.model_dump_json())


if __name__ == "__main__":
    main()
