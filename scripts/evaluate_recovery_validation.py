"""Predeclared unseen timing sweep and physical stop checks; frozen controller."""

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest
from scripts.evaluate_recovery_positives import run


def validate(config):
    values = config["stretches"]
    if (
        not values
        or len(values) != len(set(values))
        or any(type(v) not in (float, int) or not np.isfinite(v) or not 1 <= v <= 3 for v in values)
    ):
        raise ValueError("invalid timing sweep")
    if set(values) & set(config["excluded_stretches"]):
        raise ValueError("timing sweep overlaps observed values")
    if config["modes"] != ["baseline", "fixed", "descent"]:
        raise ValueError("three paired modes required")


def summarize(rows):
    result = {}
    bases = {r["stretch"]: r for r in rows if r["mode"] == "baseline"}
    for mode in ("baseline", "fixed", "descent"):
        group = [r for r in rows if r["mode"] == mode]
        result[mode] = {
            "cases": len(group),
            "successes": sum(r["quality"]["verified_pick_place"] for r in group),
            "baseline_successes": sum(
                bases[r["stretch"]]["quality"]["verified_pick_place"] for r in group
            ),
            "unnecessary_takeovers": sum(
                bases[r["stretch"]]["quality"]["verified_pick_place"]
                and r["takeover_tick"] is not None
                for r in group
            ),
            "rescued_failures": sum(
                not bases[r["stretch"]]["quality"]["verified_pick_place"]
                and r["quality"]["verified_pick_place"]
                for r in group
            ),
            "lost_successes": sum(
                bases[r["stretch"]]["quality"]["verified_pick_place"]
                and not r["quality"]["verified_pick_place"]
                for r in group
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/recovery-validation.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate(config)
    for path, expected in config["inputs_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    manifest = Path(config["manifest"])
    trace = Path(config["trace"])
    initial = json.loads(manifest.read_text())["episodes"][5]["initial_state"]
    with np.load(trace) as data:
        actions = data["actions"].copy()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        Path("scripts/evaluate_recovery_positives.py"),
        Path("scripts/audit_contact_watchdog.py"),
    ]
    sources += [
        Path("src/embodied_agent") / name
        for name in (
            "contact_watchdog.py",
            "physics.py",
            "correction_data.py",
            "grasp_quality.py",
            "assets/tabletop.xml",
        )
    ]
    report = {
        "status": "running",
        "config": config,
        "config_sha256": digest(args.config),
        "scope": "Unseen playback timing values in one known training scene; not new learned-policy or spatial generalization.",
        "source_sha256": {str(p): digest(p) for p in sources},
        "results": [],
        "stops": [],
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for stretch in config["stretches"]:
            for mode in config["modes"]:
                row = run(initial, actions, stretch, mode, output)
                report["results"].append(row)
                save()
                print(
                    row["name"],
                    row["quality"]["verified_pick_place"],
                    row["takeover_tick"],
                    flush=True,
                )
        report["scores"] = summarize(report["results"])
        report["paired_checks"] = []
        for row in report["results"]:
            if row["mode"] == "baseline":
                continue
            base = next(
                r
                for r in report["results"]
                if r["stretch"] == row["stretch"] and r["mode"] == "baseline"
            )
            with np.load(output / base["trace"]) as a, np.load(output / row["trace"]) as b:
                prefix = row["takeover_tick"]
                same = all(
                    np.array_equal(a[k][:prefix], b[k][:prefix])
                    for k in ("actions", "qpos", "tips", "holding")
                )
            report["paired_checks"].append(
                {"name": row["name"], "same_prefix_or_entire_trace": same}
            )
            if not same:
                raise RuntimeError("pair mismatch")
        for case in config["stops"]:
            row = run(initial, actions, case["stretch"], "descent", output, stop_tick=case["tick"])
            with np.load(output / row["trace"]) as data:
                tail = data["actions"][case["tick"] :]
                latched = len(tail) == 26 and bool(np.all(tail == tail[0]))
            expected = case["expected_takeover_tick"]
            row["checks"] = {
                "command_latched_26_frames": latched,
                "user_stop_reason": row["reason"] == "user stop",
                "expected_takeover": row["takeover_tick"] == expected,
                "learner_frozen": row["learner_ticks"]
                == (case["tick"] if expected is None else expected),
                "grace_observed": row["deferred_ticks"] > 0 if case["during_grace"] else True,
            }
            report["stops"].append(row)
            save()
            if not all(row["checks"].values()):
                raise RuntimeError("stop regression")
        report["trace_checks"] = []
        for row in report["results"] + report["stops"]:
            path = output / row["trace"]
            with np.load(path) as data:
                finite = bool(np.isfinite(data["actions"]).all())
            check = {
                "name": row["name"],
                "hash_matches": digest(path) == row["trace_sha256"],
                "finite": finite,
            }
            report["trace_checks"].append(check)
            if not finite or not check["hash_matches"]:
                raise RuntimeError("invalid trace")
        if any(digest(p) != expected for p, expected in report["source_sha256"].items()):
            raise RuntimeError("source changed during validation")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
