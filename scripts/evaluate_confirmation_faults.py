"""Frozen synthetic sensor-fault matrix; fixture truth never enters the runtime."""

import argparse
import hashlib
import json
from pathlib import Path

from embodied_agent.mock import MockPolicy
from embodied_agent.models import Observation, Plan, PlanStep
from embodied_agent.runtime import AgentRuntime

CASES = (
    ("stable", True),
    ("delayed", True),
    ("transient", False),
    ("stale", False),
    ("replay", False),
    ("future", False),
)
PROFILES = {
    "immediate": {},
    "fresh": {"confirmation_timeout_s": 0.15, "max_observation_age_s": 0.1},
    "temporal": {
        "confirmation_timeout_s": 0.15,
        "max_observation_age_s": 0.1,
        "confirmation_samples": 3,
        "confirmation_span_s": 0.03,
        "max_confirmation_gap_s": 0.025,
    },
}


class Planner:
    def create_plan(self, *args):
        return Plan(
            "verify destination", [PlanStep("verify", {"object": "block", "destination": "box"})]
        )


class SensorFixture:
    def __init__(self, case):
        self.case, self.time, self.reads, self.executions = case, 10.0, 0, 0
        self.first_stamp = None

    def execute(self, action):
        self.executions += 1

    def sleep(self, duration):
        self.time += duration

    def observe(self):
        if not self.executions:
            return Observation()
        self.reads += 1
        self.time += 0.001
        stamp = self.time
        if self.first_stamp is None:
            self.first_stamp = stamp
        positive = True
        if self.case == "delayed":
            positive = self.reads >= 3
        elif self.case == "transient":
            positive = self.reads == 1
        elif self.case == "stale":
            stamp = 9.0
        elif self.case == "replay":
            stamp = self.first_stamp
        elif self.case == "future":
            stamp += 1
        return Observation(
            objects={"block": "box" if positive else "table"},
            metadata={"observed_at_monotonic": stamp},
        )


def evaluate():
    results = []
    for name, options in PROFILES.items():
        for case, expected in CASES:
            fixture = SensorFixture(case)
            outcome = AgentRuntime(
                Planner(),
                MockPolicy(),
                fixture,
                clock=lambda fixture=fixture: fixture.time,
                sleep=fixture.sleep,
                **options,
            ).run("verify")
            results.append(
                {
                    "profile": name,
                    "case": case,
                    "expected_success": expected,
                    "success": outcome.success,
                    "false_accept": outcome.success and not expected,
                    "missed_valid_case": expected and not outcome.success,
                    "executions": fixture.executions,
                    "reads": fixture.reads,
                    "reports": outcome.confirmation_reports,
                    "message": outcome.message,
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/evaluate_confirmation_faults.py",
        "src/embodied_agent/runtime.py",
        "src/embodied_agent/models.py",
        "src/embodied_agent/mock.py",
        "src/embodied_agent/safety.py",
    ]

    def hashes():
        return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}

    sources = hashes()
    (out / "plan.json").write_text(
        json.dumps({"cases": CASES, "profiles": PROFILES, "sources": sources}, indent=2) + "\n"
    )
    cases = evaluate()
    summary = {
        name: {
            "cases": 6,
            "false_accepts": sum(c["false_accept"] for c in cases if c["profile"] == name),
            "missed_valid_cases": sum(
                c["missed_valid_case"] for c in cases if c["profile"] == name
            ),
        }
        for name in PROFILES
    }
    if hashes() != sources:
        raise RuntimeError("sources changed")
    report = {
        "sources": sources,
        "summary": summary,
        "cases": cases,
        "synthetic": True,
        "clock": "deterministic fixture",
        "hardware_validated": False,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if any(c["executions"] != 1 for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
