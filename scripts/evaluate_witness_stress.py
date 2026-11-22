"""Synthetic reference-source ablation, including an explicit common-mode failure."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from embodied_agent.mock import MockPolicy
from embodied_agent.models import Observation
from embodied_agent.runtime import AgentRuntime
from scripts.evaluate_confirmation_faults import PROFILES, Planner
from scripts.evaluate_confirmation_stress import Stream, cases

MODES = ("single", "ideal_reference", "noisy_reference", "common_mode")


def plan():
    result = cases()
    for case in result:
        rng = random.Random(case["seed"] + 90000)
        # An oracle is deliberately an optimistic bound, not a real second sensor.
        case["reference"] = [case["expected"]] * 64
        case["noisy_reference"] = [case["expected"] != (rng.random() < 0.1) for _ in range(64)]
    return result


class ObservedStream(Stream):
    def observe(self):
        self.last_observation = super().observe()
        return self.last_observation


class Reference:
    def __init__(self, stream, mode):
        self.stream, self.mode = stream, mode

    def __call__(self, action):
        del action
        if self.mode == "common_mode":
            # Deliberately duplicate the primary, to falsify "two callbacks = independence".
            return self.stream.last_observation
        index = min(int((self.stream.time - 10) / 0.005), 63)
        key = "reference" if self.mode == "ideal_reference" else "noisy_reference"
        positive = self.stream.case[key][index]
        return Observation(
            objects={"block": "box" if positive else "table"},
            metadata={"observed_at_monotonic": 10 + index * 0.005},
        )


def evaluate(campaign):
    results = []
    for case in campaign:
        for mode in MODES:
            stream = ObservedStream(case)
            witness = None if mode == "single" else Reference(stream, mode)
            result = AgentRuntime(
                Planner(),
                MockPolicy(),
                stream,
                clock=lambda stream=stream: stream.time,
                sleep=stream.sleep,
                confirmation_witness=witness,
                **PROFILES["temporal"],
            ).run("verify")
            results.append(
                {
                    "mode": mode,
                    "kind": case["kind"],
                    "seed": case["seed"],
                    "expected": case["expected"],
                    "success": result.success,
                    "false_accept": result.success and not case["expected"],
                    "missed_valid": case["expected"] and not result.success,
                    "executions": stream.executions,
                    "reads": stream.reads,
                    "reports": result.confirmation_reports,
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/evaluate_witness_stress.py",
        "scripts/evaluate_confirmation_stress.py",
        "scripts/evaluate_confirmation_faults.py",
        "src/embodied_agent/runtime.py",
        "src/embodied_agent/models.py",
        "src/embodied_agent/mock.py",
        "src/embodied_agent/safety.py",
    ]

    def hashes():
        return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}

    sources = hashes()
    campaign = plan()
    planfile = out / "plan.json"
    planfile.write_text(
        json.dumps(
            {
                "cases": campaign,
                "modes": MODES,
                "sources": sources,
                "temporal_profile": PROFILES["temporal"],
            },
            indent=2,
        )
        + "\n"
    )
    results = evaluate(campaign)
    summary = []
    for mode in MODES:
        rows = [r for r in results if r["mode"] == mode]
        summary.append(
            {
                "mode": mode,
                "valid_cases": 60,
                "fault_cases": 100,
                "false_accepts": sum(r["false_accept"] for r in rows),
                "missed_valid": sum(r["missed_valid"] for r in rows),
            }
        )
    report = out / "report.json"
    report.write_text(json.dumps(results, indent=2) + "\n")
    if hashes() != sources or any(r["executions"] != 1 for r in results):
        raise RuntimeError("source drift or replay")
    output = {
        "sources": sources,
        "summary": summary,
        "runs": len(results),
        "plan_path": str(planfile),
        "plan_sha256": hashlib.sha256(planfile.read_bytes()).hexdigest(),
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
    (out / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
