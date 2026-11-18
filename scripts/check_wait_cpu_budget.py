"""Fixed 216-window comparison of sleep, spin and request-boundary CPU fallback."""

import argparse
import json
from pathlib import Path

from scripts.check_evidence_checkpoint import digest
from scripts.check_fixed_wait_probe import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_wait_cpu_budget.py",
        "scripts/check_fixed_wait_probe.py",
        "scripts/check_process_pair_capture.py",
        "src/embodied_agent/scheduled_pair_capture.py",
        "src/embodied_agent/process_pair_capture.py",
        "src/embodied_agent/capture_supervisor.py",
    ]
    sources = {p: digest(p) for p in paths}
    plan = [
        ("sleep", 0, None),
        ("spin", 0.0015, None),
        ("budget", 0.0015, 0.005),
        ("budget", 0.0015, 0.005),
        ("spin", 0.0015, None),
        ("sleep", 0, None),
    ]
    (out / "plan.json").write_text(json.dumps({"plan": plan, "sources": sources}, indent=2) + "\n")
    cases, compact = [], []
    for label, margin, budget in plan:
        case = run(margin, wait_cpu_budget_s=budget)
        cases.append({"label": label, "budget_s": budget, **case})
        waits = [t["reservation_wait"] for t in case["capture"]["timing_trace"]]
        previous = 0
        valid = True
        for w in waits:
            expected = 0 if budget is not None and previous >= budget else margin
            valid = valid and w["effective_spin_window_s"] == expected
            if budget is not None:
                valid = valid and abs(w["wait_cpu_spent_s"] - previous - w["wait_cpu_s"]) < 1e-9
                previous = w["wait_cpu_spent_s"]
        fallback = [
            w["request_id"]
            for w in waits
            if budget is not None and w["effective_spin_window_s"] == 0
        ]
        compact.append(
            {
                "label": label,
                "budget_s": budget,
                **case["summary"],
                "fallback_requests": fallback,
                "policy_trace_valid": valid,
                "workers_stopped": case["capture"]["workers_stopped"],
            }
        )
        (out / f"case-{len(cases)}.json").write_text(json.dumps(cases[-1], indent=2) + "\n")
    if sources != {p: digest(p) for p in paths}:
        raise RuntimeError("sources changed")
    report = out / "report.json"
    report.write_text(json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n")
    passed = all(
        c["measurement_complete"]
        and c["workers_stopped"]
        and c["policy_trace_valid"]
        and (c["label"] != "budget" or c["fallback_requests"])
        for c in compact
    )
    summary = {
        "sources": sources,
        "report_path": str(report),
        "report_sha256": digest(report),
        "cases": compact,
        "passed": bool(passed),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(compact, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
