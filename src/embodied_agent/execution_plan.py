"""Compile a validated task to the fixed learned pick/place skill."""

from .models import PlanStep
from .safety import SafetyGate


def execution_steps(plan, execution):
    SafetyGate().validate_plan(plan)
    if execution == "scripted":
        return list(plan.steps)
    if execution != "feedback":
        raise ValueError("unknown execution mode")
    steps = list(plan.steps)
    suffix = []
    if steps and steps[-1].action == "stop":
        stop = steps.pop()
        if set(stop.arguments) - {"reason"} or not isinstance(
            stop.arguments.get("reason", ""), str
        ):
            raise ValueError("invalid stop arguments")
        suffix.append(stop)
    prefix = []
    while steps and steps[0].action == "locate":
        step = steps.pop(0)
        if step.arguments != {"object": "red_block"}:
            raise ValueError("learned execution only supports locating red_block")
        prefix.append(step)
    if (
        len(steps) != 3
        or [step.action for step in steps] != ["pick", "place", "verify"]
        or steps[0].arguments != {"object": "red_block"}
        or steps[1].arguments != {"object": "red_block", "destination": "box"}
        or steps[2].arguments
        not in (
            {"object": "red_block"},
            {"object": "red_block", "destination": None},
            {"object": "red_block", "destination": "box"},
        )
    ):
        raise ValueError(
            "学习执行仅支持定位（可选）、抓取红色积木、放入盒子、验收和末尾停止（可选）；计划未执行"
        )
    return (
        prefix
        + [
            PlanStep("learned_pick_place", {"object": "red_block", "destination": "box"}),
            steps[2],
        ]
        + suffix
    )
