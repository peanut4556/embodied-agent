"""Paired synthetic phase-latency experiment; no wall-clock performance claims."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from embodied_agent.models import Action, Observation, Plan, PlanStep
from embodied_agent.runtime import AgentRuntime

KINDS = ("fast", "planning", "selection", "execution", "confirmation", "distributed")


def case(kind, seed):
    rng = random.Random(seed)
    costs = {
        name: [rng.uniform(0.001, 0.004) for _ in range(4)]
        for name in ("selection", "execution", "confirmation")
    }
    planning = rng.uniform(0.001, 0.004)
    if kind == "planning":
        planning = rng.uniform(0.21, 0.35)
    elif kind in costs:
        costs[kind] = [rng.uniform(0.06, 0.12) for _ in range(4)]
    elif kind == "distributed":
        costs = {name: [rng.uniform(0.02, 0.04) for _ in range(4)] for name in costs}
    return {"kind": kind, "seed": seed, "planning": planning, **costs}


def run_case(spec, budget):
    class Fixture:
        time = 0.0
        index = 0
        pending = False

        def __init__(self):
            self.dispatches = []

        def create_plan(self, *args):
            self.time += spec["planning"]
            return Plan("verify", [PlanStep("verify", {"object": "b", "destination": "box"})] * 4)

        def select_action(self, step, observation):
            self.time += spec["selection"][self.index]
            return Action(step.action, step.arguments)

        def execute(self, action):
            self.dispatches.append(self.time)
            self.time += spec["execution"][self.index]
            self.pending = True

        def observe(self):
            if self.pending:
                self.time += spec["confirmation"][self.index]
                self.index += 1
                self.pending = False
            return Observation(objects={"b": "box"})

    fixture = Fixture()
    result = AgentRuntime(
        fixture, fixture, fixture, confirmation_timeout_s=0.15, clock=lambda: fixture.time
    ).run("verify", task_timeout_s=budget)
    return {
        "kind": spec["kind"],
        "seed": spec["seed"],
        "budget_s": budget,
        "success": result.success,
        "completed": result.completed_steps,
        "elapsed_s": fixture.time,
        "dispatches_s": fixture.dispatches,
        "late_dispatches": sum(t >= 0.2 for t in fixture.dispatches),
        "overshoot_s": max(0, fixture.time - 0.2),
        "message": result.message,
    }


def evaluate():
    cases = [case(kind, seed) for kind in KINDS for seed in range(31000, 31020)]
    records = [run_case(spec, budget) for spec in cases for budget in (None, 0.2)]
    summary = {}
    for budget, name in ((None, "step_only"), (0.2, "task_budget")):
        rows = [r for r in records if r["budget_s"] == budget]
        summary[name] = {
            kind: {
                "runs": len(group := [r for r in rows if r["kind"] == kind]),
                "successes": sum(r["success"] for r in group),
                "late_dispatches": sum(r["late_dispatches"] for r in group),
                "max_overshoot_s": max(r["overshoot_s"] for r in group),
            }
            for kind in KINDS
        }
    root = Path(__file__).resolve().parents[1]
    sources = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(), root / "src/embodied_agent/runtime.py")
    }
    return {
        "schema": 1,
        "synthetic": True,
        "sources_sha256": sources,
        "cases": cases,
        "records": records,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
