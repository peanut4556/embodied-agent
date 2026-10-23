"""Measure consecutive contradictions in recorded physical audit traces."""

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest


def contradiction_runs(signal, control, active):
    signal, control, active = np.asarray(signal), np.asarray(control), np.asarray(active)
    if signal.ndim != 1 or active.shape != signal.shape or control.shape != (len(signal), 5):
        raise ValueError("invalid trace shapes")
    if signal.dtype != bool or active.dtype != bool or not np.isfinite(control).all():
        raise ValueError("invalid trace values")
    counts = []
    count = 0
    for holding, command, monitoring in zip(signal, control, active, strict=True):
        count = count + 1 if monitoring and holding and np.all(command[3:] >= 0.01) else 0
        counts.append(count)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.report)
    report = json.loads(path.read_text())
    if report["status"] != "complete":
        raise ValueError("completed audit required")
    rows = []
    for row in report["results"]:
        trace = path.parent / row["trace"]
        if digest(trace) != row["trace_sha256"]:
            raise ValueError("trace hash mismatch")
        with np.load(trace) as data:
            counts = contradiction_runs(
                data["monitor_holding"], data["ctrl"], data["monitor_active"]
            )
        stops = [e["tick"] for e in row["events"] if e["event"] == "contact_inconsistent"]
        rows.append(
            {
                "name": row["name"],
                "stop_tick": stops[0] if stops else None,
                "consecutive_at_stop": counts[stops[0]] if stops else None,
                "max_consecutive": max(counts, default=0),
                "success": row["quality"]["verified_pick_place"],
                "takeover_tick": row["takeover_tick"],
            }
        )
    expected = {"healthy-release-lag-" + str(n) for n in (10, 11, 25)}
    stopped = {r["name"] for r in rows if r["stop_tick"] is not None}
    passed = stopped == expected and all(
        r["consecutive_at_stop"] == 10 for r in rows if r["name"] in expected
    )
    result = {
        "report_sha256": digest(path),
        "checker_sha256": digest(__file__),
        "rows": rows,
        "expected_limit_verified": passed,
    }
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if not passed:
        raise RuntimeError("release limit did not match frozen design")


if __name__ == "__main__":
    main()
