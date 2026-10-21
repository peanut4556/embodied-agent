"""Regression on the now-observed acquisition grid; never claim a new holdout."""

import argparse
import json
from pathlib import Path

from embodied_agent.memory_evaluation import verify_physics
from embodied_agent.memory_policy import MemoryPolicy, digest
from scripts.evaluate_acquisition_recovery import design
from scripts.evaluate_policy_generalization import paired_checks, run, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    previous = Path("outputs/evaluations/acquisition-recovery-v1")
    baseline = json.loads((previous / "report.json").read_text())
    if baseline["status"] != "complete":
        raise ValueError("completed baseline required")
    config = baseline["config"]
    policy = MemoryPolicy(config["policy"])
    if policy.metadata["weights_sha256"] != config["weights_sha256"]:
        raise ValueError("checkpoint mismatch")
    verify_physics(policy)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    # Retain exact baseline traces for independent paired checks, validating hashes.
    for row in baseline["results"]:
        if row["mode"] == "policy":
            trace = previous / row["trace"]
            if digest(trace) != row["trace_sha256"]:
                raise ValueError("baseline trace changed")
            (output / trace.name).write_bytes(trace.read_bytes())
    sources = [Path(__file__), Path("scripts/evaluate_policy_generalization.py")]
    sources += [
        Path("src/embodied_agent") / name
        for name in (
            "contact_watchdog.py",
            "correction_data.py",
            "physics.py",
            "grasp_quality.py",
            "reactive.py",
            "memory_policy.py",
            "assets/tabletop.xml",
        )
    ]
    report = {
        "status": "running",
        "scope": "Regression on 16 already observed scenes, not unseen generalization.",
        "baseline_sha256": digest(previous / "report.json"),
        "config": config,
        "source_sha256": {str(p): digest(p) for p in sources},
        "results": [],
    }
    try:
        for case in design(config):
            row = run(
                policy, case, "descent", output, 30, acquisition_seconds=4, descent_grace=True
            )
            report["results"].append(row)
            print(
                row["case_id"],
                row["quality"]["verified_pick_place"],
                row["takeover_tick"],
                flush=True,
            )
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        pairs = [r for r in baseline["results"] if r["mode"] == "policy"]
        pairs += [{**r, "mode": "guarded"} for r in report["results"]]
        report["paired_checks"] = paired_checks(pairs, output)
        report["scores"] = summarize(pairs)["guarded"]
        report["regressions_vs_fixed"] = [
            r["case_id"]
            for r in report["results"]
            if not r["quality"]["verified_pick_place"]
            and next(
                b
                for b in baseline["results"]
                if b["case_id"] == r["case_id"] and b["mode"] == "acquisition"
            )["quality"]["verified_pick_place"]
        ]
        if not all(
            c[k]
            for c in report["paired_checks"]
            for k in ("equal_initial_state", "equal_initial_image", "equal_pre_takeover_prefix")
        ):
            raise RuntimeError("baseline pairing mismatch")
        if any(digest(p) != expected for p, expected in report["source_sha256"].items()):
            raise RuntimeError("source changed during experiment")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
