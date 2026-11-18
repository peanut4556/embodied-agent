"""Two fixed supervised recovery trials with nonrefundable generation shares."""

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
        "scripts/check_recovery_wait_budget.py",
        "scripts/check_supervised_recovery.py",
        "src/embodied_agent/scheduled_pair_capture.py",
        "src/embodied_agent/capture_supervisor.py",
    ]
    sources = capture_sources() | {p: digest(p) for p in paths}
    (out / "plan.json").write_text(
        json.dumps(
            {"trials": 2, "total_threshold_s": 0.01, "generations": 2, "sources": sources}, indent=2
        )
        + "\n"
    )
    cases, summaries = [], []
    for _ in range(2):
        case = run(0.0015, True, 0.01)
        cases.append(case)
        state = case["manager"]
        waits = [row["reservation_wait"] for row in state["history"][-1]["cleanup"]["timing_trace"]]
        previous = 0
        valid = True
        for w in waits:
            valid = valid and w["wait_cpu_budget_s"] == 0.005
            valid = valid and w["effective_spin_window_s"] == (0 if previous >= 0.005 else 0.0015)
            valid = valid and abs(w["wait_cpu_spent_s"] - previous - w["wait_cpu_s"]) < 1e-9
            previous = w["wait_cpu_spent_s"]
        budget = state["wait_cpu_budget"]
        fallback = [w["request_id"] for w in waits if w["effective_spin_window_s"] == 0]
        checks = (
            case["old_timeout_observed"]
            and case["old_cleanup"]["workers_stopped"]
            and [a["allowance_s"] for a in budget["allocations"]] == [0.005, 0.005]
            and budget["unallocated_s"] == 0
            and case["generation_budget_enforced"]
            and case["old_session_attempt"].get("reason") == "session_mismatch"
            and state["state"] == "closed"
            and len(waits) == case["requests"]
            and bool(waits)
            and valid
        )
        summaries.append(
            {
                "budget_and_cleanup_verified": bool(checks),
                "full_recovery_passed": case["passed"],
                "confirmation_status": case["confirmation_status"],
                "requests": case["requests"],
                "fallback_requests": fallback,
                "wait_cpu_s": previous,
                "budget": budget,
                "receipt": case["receipt"],
                "repeat": case["repeat"],
            }
        )
        (out / f"case-{len(cases)}.json").write_text(json.dumps(case, indent=2) + "\n")
    if sources != capture_sources() | {p: digest(p) for p in paths}:
        raise RuntimeError("sources changed")
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
    if not all(c["budget_and_cleanup_verified"] for c in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
