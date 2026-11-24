"""Sequential synthetic drift experiment with sticky versus reset budgets."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from embodied_agent.action_budget import ActionBudget
from embodied_agent.models import Action, Observation, Plan, PlanStep
from embodied_agent.runtime import AgentRuntime
from scripts.evaluate_action_budget import RANGES


class Fixture:
    def __init__(self, name, duration, remaining):
        self.name, self.duration, self.remaining = name, duration, remaining
        self.t, self.calls = 0.0, 0

    def create_plan(self, *args):
        self.t += 0.1 - self.remaining
        return Plan("test", [PlanStep(self.name, {"object": "b", "destination": "box"})])

    def select_action(self, step, observation):
        return Action(step.action, step.arguments)

    def execute(self, action):
        self.calls += 1
        self.t += self.duration

    def observe(self):
        return Observation(
            objects={"b": "gripper" if self.name == "pick" else "box"},
            gripper_holding="b" if self.name == "pick" else "",
        )


def evaluate():
    calibration = {
        name: [random.Random(seed).uniform(*bounds) for seed in range(51000, 51100)]
        for name, bounds in RANGES.items()
    }
    rows = []
    for scenario in ("tail", "shift", "recovery"):
        for name, bounds in RANGES.items():
            sticky = ActionBudget(calibration)
            for index in range(40):
                rng = random.Random(71000 + index)
                duration = rng.uniform(*bounds)
                abnormal = (
                    rng.random() < 0.15
                    if scenario == "tail"
                    else index >= 20
                    if scenario == "shift"
                    else 10 <= index < 20
                )
                if abnormal:
                    duration *= 4
                remaining = rng.uniform(0.01, 0.08)
                for mode in ("fixed", "reset", "sticky"):
                    budget = sticky if mode == "sticky" else ActionBudget(calibration)
                    before = name in budget.degraded
                    required = 0.05 if mode == "fixed" else budget.required(name)
                    fixture = Fixture(name, duration, remaining)
                    result = AgentRuntime(
                        fixture, fixture, fixture, clock=lambda fixture=fixture: fixture.t
                    ).run(
                        "test",
                        task_timeout_s=0.1,
                        minimum_step_budget_s=0.05 if mode == "fixed" else 0,
                        action_budget=None if mode == "fixed" else budget,
                    )
                    rows.append(
                        {
                            "scenario": scenario,
                            "action": name,
                            "index": index,
                            "mode": mode,
                            "duration_s": duration,
                            "remaining_s": remaining,
                            "abnormal": abnormal,
                            "required_s": required,
                            "degraded_before": before,
                            "degraded_after": name in budget.degraded,
                            "feasible": duration < remaining,
                            "success": result.success,
                            "dispatched": fixture.calls,
                            "unconfirmed": fixture.calls - result.completed_steps,
                        }
                    )
    summary = {}
    for mode in ("fixed", "reset", "sticky"):
        group = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "runs": len(group),
            "successes": sum(r["success"] for r in group),
            "rejected_feasible": sum(r["feasible"] and not r["success"] for r in group),
            "unconfirmed": sum(r["unconfirmed"] for r in group),
            "unconfirmed_after_degradation": sum(
                r["unconfirmed"] for r in group if r["degraded_before"]
            ),
        }
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "src/embodied_agent/action_budget.py",
        root / "src/embodied_agent/runtime.py",
        root / "scripts/evaluate_action_budget.py",
    ]
    return {
        "calibration": calibration,
        "rows": rows,
        "summary": summary,
        "sources_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate()
    with args.output.open("x") as stream:
        json.dump(report, stream, separators=(",", ":"))
    print(json.dumps(report["summary"], indent=2))
