"""Verify recorded coverage for each scenario in a follow-up stress report."""

import argparse
import json
from collections import Counter
from pathlib import Path

from embodied_agent.coverage_audit import audit_coverage
from embodied_agent.recovery_report import write_recovery_report


def audit_matrix(report):
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise TypeError("stress report with results required")
    if not report["results"]:
        raise ValueError("empty stress report")
    rows = []
    for index, row in enumerate(report["results"]):
        try:
            if not isinstance(row, dict) or "actual_status" not in row:
                raise ValueError("scenario status required")
            audit = audit_coverage(row.get("coverage"), status=row["actual_status"])
        except ValueError as exc:
            audit = {"status": "rejected", "error": str(exc)}
        rows.append({"index": index, **audit})
    return {
        "schema_version": 1,
        "scope": "coverage_arithmetic_only",
        "counts": dict(Counter(row["status"] for row in rows)),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_matrix(json.loads(Path(args.input).read_text(encoding="utf-8")))
    write_recovery_report(args.output, report)
    print(report["counts"])
    return (
        1 if report["counts"].get("rejected") else 2 if report["counts"].get("unavailable") else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
