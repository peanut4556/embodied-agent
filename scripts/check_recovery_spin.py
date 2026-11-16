"""Counterbalanced sleep versus final-window spin after supervised recovery."""

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
        "scripts/check_recovery_spin.py",
        "scripts/check_supervised_recovery.py",
        "src/embodied_agent/scheduled_pair_capture.py",
        "src/embodied_agent/capture_supervisor.py",
    ]
    sources = capture_sources() | {p: digest(p) for p in paths}
    cases = [run(margin) for margin in (0, 0.0015, 0.0015, 0)]
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
    print([(c["spin_window_s"], c["passed"], c["requests"]) for c in cases])


if __name__ == "__main__":
    main()
