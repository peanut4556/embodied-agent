"""Paired seeded synthetic stress campaign; persistent false sensors remain a counterexample."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from embodied_agent.mock import MockPolicy
from embodied_agent.models import Observation
from embodied_agent.runtime import AgentRuntime
from scripts.evaluate_confirmation_faults import PROFILES, Planner

KINDS = ("stable", "delayed", "dropout", "pulse", "burst", "stale", "replay", "persistent_false")


def cases():
    output = []
    for kind in KINDS:
        for repeat in range(20):
            seed = 22000 + repeat
            rng = random.Random(seed)
            delay, width = rng.uniform(0.01, 0.12), rng.uniform(0.005, 0.08)
            frames = []
            last = 0
            for index in range(64):
                t = index * 0.005
                positive = (
                    t >= delay
                    if kind == "delayed"
                    else t <= width
                    if kind in ("pulse", "burst")
                    else True
                )
                if kind == "pulse":
                    positive = t <= 0.005
                if kind == "dropout" and rng.random() < 0.45:
                    frames.append(
                        frames[last].copy() if frames else {"stamp": 0, "positive": False}
                    )
                    continue
                last = index
                frames.append({"stamp": t, "positive": positive})
            output.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "expected": kind in ("stable", "delayed", "dropout"),
                    "frames": frames,
                    "read_costs": [rng.uniform(0.001, 0.006) for _ in range(128)],
                }
            )
    return output


class Stream:
    def __init__(self, case):
        self.case, self.time, self.reads, self.executions = case, 10.0, 0, 0

    def execute(self, action):
        self.executions += 1

    def sleep(self, delay):
        self.time += delay

    def observe(self):
        if not self.executions:
            return Observation()
        self.time += self.case["read_costs"][self.reads]
        self.reads += 1
        frame = self.case["frames"][min(int((self.time - 10) / 0.005), 63)]
        stamp = 10 + frame["stamp"]
        if self.case["kind"] == "stale":
            stamp = 9
        elif self.case["kind"] == "replay":
            stamp = 10
        return Observation(
            objects={"block": "box" if frame["positive"] else "table"},
            metadata={"observed_at_monotonic": stamp},
        )


def evaluate(plan):
    rows = []
    for case in plan:
        for profile, options in PROFILES.items():
            stream = Stream(case)
            result = AgentRuntime(
                Planner(),
                MockPolicy(),
                stream,
                clock=lambda stream=stream: stream.time,
                sleep=stream.sleep,
                **options,
            ).run("verify")
            rows.append(
                {
                    "kind": case["kind"],
                    "seed": case["seed"],
                    "profile": profile,
                    "expected": case["expected"],
                    "success": result.success,
                    "false_accept": result.success and not case["expected"],
                    "missed_valid": not result.success and case["expected"],
                    "executions": stream.executions,
                    "reads": stream.reads,
                    "report": result.confirmation_reports,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/evaluate_confirmation_stress.py",
        "scripts/evaluate_confirmation_faults.py",
        "src/embodied_agent/runtime.py",
        "src/embodied_agent/models.py",
        "src/embodied_agent/safety.py",
        "src/embodied_agent/mock.py",
    ]

    def hashes():
        return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}

    sources = hashes()
    plan = cases()
    planfile = out / "plan.json"
    planfile.write_text(
        json.dumps({"cases": plan, "profiles": PROFILES, "sources": sources}, indent=2) + "\n"
    )
    rows = evaluate(plan)
    summary = []
    for profile in PROFILES:
        for kind in KINDS:
            selected = [r for r in rows if r["profile"] == profile and r["kind"] == kind]
            summary.append(
                {
                    "profile": profile,
                    "kind": kind,
                    "cases": len(selected),
                    "false_accepts": sum(r["false_accept"] for r in selected),
                    "missed_valid": sum(r["missed_valid"] for r in selected),
                }
            )
    report = out / "report.json"
    report.write_text(json.dumps(rows, indent=2) + "\n")
    if hashes() != sources or any(r["executions"] != 1 for r in rows):
        raise RuntimeError("source drift or action replay")
    result = {
        "sources": sources,
        "summary": summary,
        "runs": len(rows),
        "plan_path": str(planfile),
        "plan_sha256": hashlib.sha256(planfile.read_bytes()).hexdigest(),
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
