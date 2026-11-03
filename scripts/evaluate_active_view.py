"""Paired physical playback audit of failure-triggered calibrated view selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest
from scripts.evaluate_release_observer import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path("config/contact-delayed-release.json")
    config = json.loads(config_path.read_text())
    for path, expected in config["inputs_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"historical input changed: {path}")
    initial = json.loads(Path(config["manifest"]).read_text())["episodes"][5]["initial_state"]
    with np.load(config["sources"][0]["trace"]) as data:
        actions = data["actions"].copy()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), config_path]
    paths += [
        Path("scripts") / name
        for name in (
            "evaluate_release_observer.py",
            "audit_contact_watchdog.py",
            "evaluate_contact_faults.py",
            "evaluate_recovery_positives.py",
        )
    ]
    paths += [
        Path("src/embodied_agent") / name
        for name in (
            "active_view.py",
            "release_observer.py",
            "release_confirmation.py",
            "release_review.py",
            "contact_watchdog.py",
            "physics.py",
            "perception.py",
            "grasp_quality.py",
            "assets/tabletop.xml",
        )
    ]
    hashes = {str(path): digest(path) for path in paths}
    report = {
        "schema_version": 1,
        "status": "running",
        "source_sha256": hashes,
        "inputs_sha256": config["inputs_sha256"],
        "scope": "Known-scene full physical playback, rendered RGB-D, virtual calibrated views. "
        "No training, no arm recovery, no unseen-scene claim.",
        "cases": [],
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for name, lag, missing in (
            ("nominal", 0, "none"),
            ("lag9", 9, "none"),
            ("early_missing", 0, "early"),
            ("all_missing", 0, "all"),
        ):
            rows = {}
            for view in ("original", "offset", "active"):
                rows[view] = run(
                    initial, actions, lag, missing, True, output / f"{name}-{view}.npz", view=view
                )
                print(name, view, rows[view]["confirmation"], flush=True)
            with np.load(output / rows["original"]["trace"]) as baseline:
                equal = True
                for view in ("offset", "active"):
                    with np.load(output / rows[view]["trace"]) as candidate:
                        equal = equal and all(
                            np.array_equal(baseline[key], candidate[key])
                            for key in ("qpos", "actions")
                        )
            same_quality = all(
                row["quality"] == rows["original"]["quality"] for row in rows.values()
            )
            expected = "unconfirmed" if missing == "all" else "confirmed_in_tray"
            report["cases"].append(
                {
                    "name": name,
                    "identical_execution": equal,
                    "identical_quality": same_quality,
                    "expected_active": expected,
                    "matches_expected": rows["active"]["confirmation"] == expected,
                    "views": rows,
                }
            )
            save()
        if hashes != {str(path): digest(path) for path in paths}:
            raise RuntimeError("source changed during evaluation")
        report["confirmations"] = {
            view: sum(
                case["views"][view]["confirmation"] == "confirmed_in_tray"
                for case in report["cases"]
            )
            for view in ("original", "offset", "active")
        }
        report["passed"] = all(
            case["identical_execution"] and case["identical_quality"] and case["matches_expected"]
            for case in report["cases"]
        )
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
