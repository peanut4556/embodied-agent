"""Synthetic admission threshold trade-offs; truth is used only for scoring."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from embodied_agent.models import Action, Observation, Plan, PlanStep
from embodied_agent.runtime import AgentRuntime
from scripts.evaluate_task_budget import case

RESERVES = (0, 0.02, 0.05, 0.1, 0.2)


def run_case(spec, reserve):
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
    ).run("verify", task_timeout_s=0.2, minimum_step_budget_s=reserve)
    return {
        "kind": spec["kind"],
        "seed": spec["seed"],
        "reserve_s": reserve,
        "success": result.success,
        "completed": result.completed_steps,
        "elapsed_s": fixture.time,
        "dispatches_s": fixture.dispatches,
        "late_dispatches": sum(t >= 0.2 for t in fixture.dispatches),
        "overshoot_s": max(0, fixture.time - 0.2),
        "message": result.message,
        "unconfirmed_actions": len(fixture.dispatches) - result.completed_steps,
    }


def evaluate():
    cases = []
    for kind in ("fast", "selection", "execution", "confirmation"):
        for seed in range(41000, 41020):
            spec = case(kind, seed)
            total = spec["planning"] + sum(
                sum(spec[k]) for k in ("selection", "execution", "confirmation")
            )
            target = random.Random(seed).uniform(0.12, 0.28)
            scale = target / total
            spec["planning"] *= scale
            for key in ("selection", "execution", "confirmation"):
                spec[key] = [v * scale for v in spec[key]]
            spec["feasible"] = target < 0.2
            cases.append(spec)
    records = []
    for spec in cases:
        for reserve in RESERVES:
            row = run_case(spec, reserve)
            row["feasible"] = spec["feasible"]
            records.append(row)
    summary = {}
    for reserve in RESERVES:
        rows = [r for r in records if r["reserve_s"] == reserve]
        summary[str(reserve)] = {
            "runs": len(rows),
            "feasible_cases": sum(r["feasible"] for r in rows),
            "successes": sum(r["success"] for r in rows),
            "rejected_feasible": sum(r["feasible"] and not r["success"] for r in rows),
            "unconfirmed_actions": sum(r["unconfirmed_actions"] for r in rows),
            "late_dispatches": sum(r["late_dispatches"] for r in rows),
            "max_overshoot_s": max(r["overshoot_s"] for r in rows),
        }
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "scripts/evaluate_task_budget.py",
        root / "src/embodied_agent/runtime.py",
    )
    return {
        "schema": 1,
        "synthetic": True,
        "sources_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
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
        json.dump(report, stream, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
