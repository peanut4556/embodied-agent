"""Measure waiting-thread CPU cost and target lateness in recovery A/B/B/A runs."""

import argparse
import json
from pathlib import Path

from scripts.check_evidence_checkpoint import digest
from scripts.check_recovery_handoff import sources as capture_sources
from scripts.check_supervised_recovery import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_wait_cost.py",
        "scripts/check_supervised_recovery.py",
        "src/embodied_agent/scheduled_pair_capture.py",
        "src/embodied_agent/capture_supervisor.py",
    ]
    sources = capture_sources() | {p: digest(p) for p in paths}
    cases = [run(margin, trace_timing=True) for margin in (0, 0.0015, 0.0015, 0)]
    summaries = []
    for case in cases:
        runtime = case["manager"]["history"][-1]["result"]
        trace = runtime["capture"]["timing_trace"]
        waits = [row["reservation_wait"] for row in trace if row.get("reservation_wait")]
        summaries.append(
            {
                "spin_window_s": case["spin_window_s"],
                "passed": case["passed"],
                "requests": case["requests"],
                "measured_requests": len(waits),
                "wait_cpu_s": sum(w["wait_cpu_s"] for w in waits),
                "wait_elapsed_s": sum(w["wait_elapsed_s"] for w in waits),
                "max_target_lateness_s": max((w["target_lateness_s"] for w in waits), default=None),
                "missed_windows": sum(w["window_missed"] for w in waits),
                "sleep_calls": sum(w["sleep_calls"] for w in waits),
                "spin_iterations": sum(w["spin_iterations"] for w in waits),
                "reason": runtime["confirmation"]["reason"],
            }
        )
    assert sources == capture_sources() | {p: digest(p) for p in paths}
    report = out / "report.json"
    report.write_text(json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n")
    summary = {
        "sources": sources,
        "report_path": str(report),
        "report_sha256": digest(report),
        "cases": summaries,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
