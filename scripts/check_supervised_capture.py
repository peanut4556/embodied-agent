"""Full reserved confirmation through independently supervised paired readers."""

import argparse
import json
from pathlib import Path

from embodied_agent.capture_recovery import CaptureRecovery
from scripts.check_evidence_checkpoint import digest
from scripts.check_process_pair_capture import fast_factory
from scripts.check_recovery_handoff import drive
from scripts.check_recovery_handoff import sources as capture_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = ["scripts/check_supervised_capture.py", "src/embodied_agent/capture_supervisor.py"]
    sources = capture_sources() | {p: digest(p) for p in paths}
    cases = []
    for _ in range(2):
        manager = CaptureRecovery(
            position_error_m=(0.001,) * 3, capture_profile="reserved", supervised=True
        )
        try:
            token = manager.start(fast_factory)
            if token:
                drive(manager, token)
            status = manager.runtime.controller.status if manager.runtime else "startup_failed"
            requests = len(manager.runtime.controller.requests) if manager.runtime else 0
        finally:
            closed = manager.close()
        cases.append(
            {
                "status": status,
                "requests": requests,
                "closed": closed,
                "passed": status == "confirmed_recent_evidence" and closed,
                "manager": manager.result(),
            }
        )
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
    print(summary["cases"])
    if not all(c["passed"] for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
