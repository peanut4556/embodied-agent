"""Explicit session recovery after independently bounded initialization failure."""

import argparse
import json
import signal
import time
from pathlib import Path

from embodied_agent.capture_recovery import CaptureRecovery
from scripts.check_evidence_checkpoint import digest
from scripts.check_process_pair_capture import fast_factory
from scripts.check_recovery_handoff import drive
from scripts.check_recovery_handoff import sources as capture_sources


def blocked_factory():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(0.01)


def run():
    manager = CaptureRecovery(
        position_error_m=(0.001,) * 3,
        max_generations=2,
        capture_profile="reserved",
        supervised=True,
        runtime_limit_s=4,
    )
    result = {}
    try:
        first = manager.start(blocked_factory)
        old_session = manager.session
        old_backend = manager.backend.result()
        timeout_seen = any(
            d["events"] and d["events"][-1].get("runtime_limit_exceeded")
            for d in old_backend["supervision"]
        )
        token = manager.start(fast_factory)
        stale = manager.consume_confirmation(old_session)
        if token:
            drive(manager, token)
            receipt = manager.consume_confirmation(token)
            duplicate = manager.consume_confirmation(token)
            confirmation = manager.runtime.controller.result()
        else:
            receipt = duplicate = {"status": "not_attempted"}
            confirmation = {"status": "startup_failed", "requests": []}
        closed = manager.close()
        try:
            manager.start(fast_factory)
        except RuntimeError as exc:
            exhausted = str(exc) == "recovery generation budget exhausted"
        else:
            exhausted = False
        passed = (
            first is None
            and timeout_seen
            and old_backend["workers_stopped"]
            and token is not None
            and token != old_session
            and stale.get("reason") == "session_mismatch"
            and receipt["status"] == "consumed"
            and duplicate.get("reason") == "already_consumed"
            and confirmation["status"] == "confirmed_recent_evidence"
            and closed
            and exhausted
        )
        result = {
            "passed": passed,
            "old_timeout_observed": timeout_seen,
            "old_cleanup": old_backend,
            "old_session_attempt": stale,
            "confirmation_status": confirmation["status"],
            "requests": len(confirmation["requests"]),
            "receipt": {k: v for k, v in receipt.items() if k != "evidence"},
            "repeat": duplicate,
            "generation_budget_enforced": exhausted,
            "manager": manager.result(),
        }
    finally:
        manager.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = ["scripts/check_supervised_recovery.py", "src/embodied_agent/capture_supervisor.py"]
    sources = capture_sources() | {p: digest(p) for p in paths}
    cases = [run() for _ in range(2)]
    assert sources == capture_sources() | {p: digest(p) for p in paths}
    report = out / "report.json"
    report.write_text(json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n")
    summary = {
        "sources": sources,
        "report_path": str(report),
        "report_sha256": digest(report),
        "cases": [{k: v for k, v in c.items() if k != "manager"} for c in cases],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print([(c["passed"], c["confirmation_status"], c["requests"]) for c in cases])
    if not all(c["passed"] for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
