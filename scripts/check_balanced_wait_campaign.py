"""Predeclared, position-balanced waiting campaign; descriptive evidence only."""

import argparse
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path

from scripts.check_evidence_checkpoint import digest
from scripts.check_fixed_wait_probe import run, summarize
from scripts.check_wait_under_load import BoundedLoad

CONDITIONS = ((False, 0), (False, 0.0015), (True, 0), (True, 0.0015))


def plan():
    # Balanced Latin square: each condition occupies each position once and
    # each directed predecessor pair occurs once within blocks.
    return [
        {
            "block": block,
            "position": position,
            "loaded": CONDITIONS[c][0],
            "spin_window_s": CONDITIONS[c][1],
        }
        for block, order in enumerate(((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1)), 1)
        for position, c in enumerate(order, 1)
    ]


def load_ok(load):
    ready = [e["pid"] for e in load["events"] if e["event"] == "ready"]
    stopped = [e for e in load["events"] if e["event"] == "stopped"]
    return (
        len(ready) == len(set(ready)) == 2
        and len(stopped) == 2
        and {e["pid"] for e in stopped} == set(ready)
        and all(e["reason"] == "requested" for e in stopped)
        and load["exitcodes"] == [0, 0]
    )


def analyze(cases):
    expected = plan()
    if len(cases) != len(expected) or any(c["plan"] != p for c, p in zip(cases, expected)):
        raise ValueError("campaign must match the entire predeclared ordered plan")
    rows = []
    for case in cases:
        p, capture = case["plan"], case["capture"]
        if case["spin_window_s"] != p["spin_window_s"]:
            raise ValueError("strategy differs from plan")
        s = summarize(case["rows"], capture["timing_trace"], 36)
        valid = s["measurement_complete"] and capture["workers_stopped"]
        valid = valid and (load_ok(case["load"]) if p["loaded"] else case["load"] is None)
        rows.append({**p, **s, "valid": bool(valid)})
    valid = all(r["valid"] for r in rows)
    groups = []
    for loaded, spin in CONDITIONS:
        group = [r for r in rows if r["loaded"] == loaded and r["spin_window_s"] == spin]
        counts = Counter()
        for r in group:
            counts.update(r["outcomes"])
        groups.append(
            {
                "loaded": loaded,
                "spin_window_s": spin,
                "planned": 144,
                "outcomes": dict(counts),
                "mean_wait_cpu_s": sum(r["wait_cpu_s"] for r in group) / 144 if valid else None,
                "max_target_lateness_s": max(r["max_target_lateness_s"] for r in group)
                if valid
                else None,
            }
        )
    paired = []
    if valid:
        for block in range(1, 5):
            for loaded in (False, True):
                pair = sorted(
                    [r for r in rows if r["block"] == block and r["loaded"] == loaded],
                    key=lambda r: r["spin_window_s"],
                )
                a, b = pair
                paired.append(
                    {
                        "block": block,
                        "loaded": loaded,
                        "spin_minus_sleep_delivered": b["outcomes"].get("delivered", 0)
                        - a["outcomes"].get("delivered", 0),
                        "spin_minus_sleep_wait_cpu_s": b["mean_wait_cpu_s"] - a["mean_wait_cpu_s"],
                    }
                )
    return {
        "campaign_valid": valid,
        "planned": 576,
        "groups": groups,
        "paired_blocks": paired,
        "cases": rows,
        "automatic_enable": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_balanced_wait_campaign.py",
        "scripts/check_wait_under_load.py",
        "scripts/check_fixed_wait_probe.py",
        "scripts/check_process_pair_capture.py",
        "src/embodied_agent/process_pair_capture.py",
        "src/embodied_agent/capture_supervisor.py",
        "src/embodied_agent/scheduled_pair_capture.py",
    ]
    sources = {p: digest(p) for p in paths}
    manifest = {
        "plan": plan(),
        "sources": sources,
        "planned": 576,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "logical_cpus": os.cpu_count(),
        },
        "load_workers": 2,
        "load_lifetime_s": 20,
    }
    (out / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cases = []
    for p in manifest["plan"]:
        load_report = None
        if p["loaded"]:
            with BoundedLoad() as load:
                result = run(p["spin_window_s"])
            load_report = {"events": load.events, "exitcodes": load.exitcodes}
        else:
            result = run(p["spin_window_s"])
        case = {"plan": p, **result, "load": load_report}
        cases.append(case)
        # Preserve completed cases even if a later case aborts unexpectedly.
        (out / f"case-{len(cases):02d}.json").write_text(json.dumps(case, indent=2) + "\n")
    if sources != {p: digest(p) for p in paths}:
        raise RuntimeError("sources changed during campaign")
    report = out / "report.json"
    report.write_text(json.dumps({**manifest, "cases": cases}, indent=2) + "\n")
    summary = {
        **analyze(cases),
        "sources": sources,
        "environment": manifest["environment"],
        "plan_sha256": digest(out / "plan.json"),
        "report_path": str(report),
        "report_sha256": digest(report),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: summary[k] for k in ("campaign_valid", "groups", "paired_blocks")}, indent=2)
    )
    if not summary["campaign_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
