"""Frozen three-arm audit of first-grasp deadline recovery, never used by default UI."""

import argparse
import json
from pathlib import Path

from embodied_agent.memory_evaluation import verify_physics
from embodied_agent.memory_policy import MemoryPolicy, digest
from scripts.evaluate_policy_generalization import cases, paired_checks, run, summarize


def design(config):
    if (
        config["modes"] != ["policy", "guarded", "acquisition"]
        or config["acquisition_seconds"] != 4
    ):
        raise ValueError("frozen three-arm design and four-second deadline required")
    return cases({**config, "modes": ["policy", "guarded"]})


def comparisons(rows, output):
    result = {}
    for mode in ("guarded", "acquisition"):
        selected = [
            {**row, "mode": "guarded" if row["mode"] == mode else "policy"}
            for row in rows
            if row["mode"] in ("policy", mode)
        ]
        checks = paired_checks(selected, output)
        result[mode] = {
            "checks": checks,
            "scores": summarize(selected)["guarded"],
            "rescued_vs_policy": sum(
                not next(
                    r for r in rows if r["case_id"] == row["case_id"] and r["mode"] == "policy"
                )["quality"]["verified_pick_place"]
                and row["quality"]["verified_pick_place"]
                for row in rows
                if row["mode"] == mode
            ),
            "baseline_success_lost": sum(c["baseline_success_lost"] for c in checks),
            "unnecessary_takeovers": sum(
                row["takeover_tick"] is not None
                and next(
                    r for r in rows if r["case_id"] == row["case_id"] and r["mode"] == "policy"
                )["quality"]["verified_pick_place"]
                for row in rows
                if row["mode"] == mode
            ),
        }
    result["policy"] = summarize(rows)["policy"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/acquisition-recovery.json")
    parser.add_argument("--output", default="outputs/evaluations/acquisition-recovery-v1")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    grid = design(config)
    for path, expected in config["inventory_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"position inventory changed: {path}")
    policy = MemoryPolicy(config["policy"])
    if policy.metadata["weights_sha256"] != config["weights_sha256"]:
        raise ValueError("checkpoint changed")
    verify_physics(policy)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path("scripts/evaluate_policy_generalization.py")]
    sources += [
        Path("src/embodied_agent") / name
        for name in (
            "contact_watchdog.py",
            "correction_data.py",
            "grasp_quality.py",
            "physics.py",
            "reactive.py",
            "memory_policy.py",
            "assets/tabletop.xml",
        )
    ]
    report = {
        "status": "running",
        "config": config,
        "config_sha256": digest(args.config),
        "source_sha256": {str(p): digest(p) for p in sources},
        "results": [],
        "limitations": "Four correlated positions x four visuals; descriptive simulation results. "
        "Hybrid expert recovery, not learner improvement. Correct camera calibration; no retraining.",
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for case in grid:
            for mode in config["modes"]:
                row = run(
                    policy,
                    case,
                    mode,
                    output,
                    config["max_seconds"],
                    acquisition_seconds=4 if mode == "acquisition" else None,
                )
                report["results"].append(row)
                save()
                print(
                    json.dumps(
                        {k: row[k] for k in ("case_id", "mode", "failure", "takeover_tick")}
                    ),
                    flush=True,
                )
        report["comparisons"] = comparisons(report["results"], output)
        if not all(
            check[key]
            for mode in ("guarded", "acquisition")
            for check in report["comparisons"][mode]["checks"]
            for key in ("equal_initial_state", "equal_initial_image", "equal_pre_takeover_prefix")
        ):
            raise RuntimeError("paired provenance mismatch")
        if any(digest(p) != expected for p, expected in report["source_sha256"].items()):
            raise RuntimeError("source changed during evaluation")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
