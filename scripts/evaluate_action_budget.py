"""Independent synthetic calibration and validation of per-action admission."""

import json
import random
from pathlib import Path

from embodied_agent.action_budget import ActionBudget
from embodied_agent.models import Action, Observation, Plan, PlanStep
from embodied_agent.runtime import AgentRuntime

RANGES = {"pick": (0.03, 0.045), "place": (0.02, 0.035), "verify": (0.005, 0.012)}


def evaluate():
    calibration = {
        name: [random.Random(seed).uniform(*bounds) for seed in range(51000, 51100)]
        for name, bounds in RANGES.items()
    }
    rows = []
    for name, bounds in RANGES.items():
        for seed in range(61000, 61040):
            rng = random.Random(seed)
            duration, remaining = rng.uniform(*bounds), rng.uniform(0.01, 0.08)
            for mode in ("fixed", "calibrated"):

                class Fixture:
                    def __init__(self, name, remaining, duration):
                        self.name, self.remaining, self.duration = name, remaining, duration

                    t = 0.0
                    calls = 0

                    def create_plan(self, *args):
                        self.t += 0.1 - self.remaining
                        return Plan(
                            "test", [PlanStep(self.name, {"object": "b", "destination": "box"})]
                        )

                    def select_action(self, step, obs):
                        return Action(step.action, step.arguments)

                    def execute(self, action):
                        self.calls += 1
                        self.t += self.duration

                    def observe(self):
                        return Observation(
                            objects={"b": "gripper" if self.name == "pick" else "box"},
                            gripper_holding="b" if self.name == "pick" else "",
                        )

                fixture = Fixture(name, remaining, duration)
                result = AgentRuntime(
                    fixture, fixture, fixture, clock=lambda fixture=fixture: fixture.t
                ).run(
                    "test",
                    task_timeout_s=0.1,
                    minimum_step_budget_s=0.05 if mode == "fixed" else 0,
                    action_budget=ActionBudget(calibration) if mode == "calibrated" else None,
                )
                rows.append(
                    {
                        "action": name,
                        "seed": seed,
                        "mode": mode,
                        "duration_s": duration,
                        "remaining_s": remaining,
                        "feasible": duration < remaining,
                        "success": result.success,
                        "dispatched": fixture.calls,
                        "unconfirmed": fixture.calls - result.completed_steps,
                    }
                )
    summary = {
        mode: {
            "runs": len(group := [r for r in rows if r["mode"] == mode]),
            "successes": sum(r["success"] for r in group),
            "feasible": sum(r["feasible"] for r in group),
            "rejected_feasible": sum(r["feasible"] and not r["success"] for r in group),
            "unconfirmed": sum(r["unconfirmed"] for r in group),
        }
        for mode in ("fixed", "calibrated")
    }
    return {"calibration": calibration, "rows": rows, "summary": summary}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate()
    with args.output.open("x") as stream:
        json.dump(report, stream, separators=(",", ":"))
    print(json.dumps(report["summary"], indent=2))
