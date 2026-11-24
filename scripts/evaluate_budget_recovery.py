"""Sequential synthetic recovery with disjoint calibration, validation and task seeds."""

import argparse
import json
import random
from pathlib import Path

from embodied_agent.action_budget import ActionBudget
from embodied_agent.runtime import AgentRuntime
from scripts.evaluate_budget_drift import Fixture


def samples(start, low, high):
    return [random.Random(seed).uniform(low, high) for seed in range(start, start + 40)]


def evaluate():
    rows, attempts = [], []
    for scenario in ("recovered", "still_slow", "second_drift"):
        for mode in ("paused", "direct", "validated"):
            budget = ActionBudget(
                {"verify": samples(81000, 0.005, 0.012)}, drift_policy="quarantine"
            )
            for index in range(40):
                # Fixed maintenance opportunity, not triggered by future task labels.
                if index == 10 and mode != "paused" and budget.blocked("verify"):
                    calibration = samples(82000, 0.005, 0.012)
                    validation = (
                        samples(83000, 0.03, 0.05)
                        if scenario == "still_slow"
                        else samples(83000, 0.005, 0.012)
                    )
                    accepted = True
                    try:
                        if mode == "validated":
                            budget = budget.validated_recalibrated(
                                "verify",
                                calibration,
                                batch_id="maintenance",
                                validation_samples=validation,
                                validation_batch_id="holdout",
                            )
                        else:
                            budget = budget.recalibrated(
                                "verify", calibration, batch_id="maintenance"
                            )
                    except ValueError:
                        accepted = False
                    attempts.append(
                        {
                            "scenario": scenario,
                            "mode": mode,
                            "accepted": accepted,
                            "calibration": calibration,
                            "validation": validation,
                        }
                    )
                rng = random.Random(84000 + index)
                slow = (
                    index < 10
                    or scenario == "still_slow"
                    or (scenario == "second_drift" and index >= 25)
                )
                duration = rng.uniform(0.03, 0.05) if slow else rng.uniform(0.005, 0.012)
                remaining = rng.uniform(0.02, 0.06)
                fixture = Fixture("verify", duration, remaining)
                result = AgentRuntime(
                    fixture, fixture, fixture, clock=lambda fixture=fixture: fixture.t
                ).run("test", task_timeout_s=0.1, action_budget=budget)
                rows.append(
                    {
                        "scenario": scenario,
                        "mode": mode,
                        "index": index,
                        "duration_s": duration,
                        "remaining_s": remaining,
                        "success": result.success,
                        "feasible": duration < remaining,
                        "unconfirmed": fixture.calls - result.completed_steps,
                        "dispatched": fixture.calls,
                        "revision": budget.revision,
                        "blocked_after": budget.blocked("verify"),
                    }
                )
    summary = {}
    for mode in ("paused", "direct", "validated"):
        group = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "successes": sum(r["success"] for r in group),
            "unconfirmed": sum(r["unconfirmed"] for r in group),
            "rejected_feasible": sum(r["feasible"] and not r["success"] for r in group),
        }
    return {"rows": rows, "attempts": attempts, "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate()
    with args.output.open("x") as stream:
        json.dump(report, stream, separators=(",", ":"))
    print(json.dumps(report["summary"], indent=2))
