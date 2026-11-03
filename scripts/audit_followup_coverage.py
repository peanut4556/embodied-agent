"""Verify recorded coverage for each scenario in a follow-up stress report."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from embodied_agent.coverage_audit import audit_coverage
from embodied_agent.followup_matrix_audit import audit_membership
from embodied_agent.recovery_report import write_recovery_report


def load_report(path):
    """Reject ambiguous objects and nonfinite numbers before auditing any rows."""

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(raw):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value

    def reject_constant(raw):
        raise ValueError(f"invalid JSON constant: {raw}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=object_pairs,
        parse_float=finite_float,
        parse_constant=reject_constant,
    )


def audit_matrix(report):
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise TypeError("stress report with results required")
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("unsupported stress report schema")
    if report.get("test_kind") != "deterministic_session_fault_matrix":
        raise ValueError("unexpected stress report kind")
    if type(report.get("cases")) is not int or report["cases"] != len(report["results"]):
        raise ValueError("declared case count does not match results")
    if not report["results"]:
        raise ValueError("empty stress report")
    rows = []
    for index, row in enumerate(report["results"]):
        try:
            if not isinstance(row, dict) or row.get("actual_status") not in (
                "completed",
                "cancelled",
                "invalidated",
            ):
                raise ValueError("terminal scenario status required")
            audit = audit_coverage(row.get("coverage"), status=row["actual_status"])
        except ValueError as exc:
            audit = {"status": "rejected", "error": str(exc)}
        rows.append({"index": index, **audit})
    membership = audit_membership(report["results"])
    counts = dict(Counter(row["status"] for row in rows))
    status = (
        "rejected"
        if membership["status"] == "rejected" or counts.get("rejected")
        else "unavailable"
        if counts.get("unavailable")
        else "verified"
    )
    return {
        "schema_version": 2,
        "scope": "coverage_arithmetic_and_matrix_membership",
        "status": status,
        "matrix": membership,
        "counts": counts,
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = audit_matrix(load_report(args.input))
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(f"coverage audit input rejected: {exc}", file=sys.stderr)
        return 3
    try:
        write_recovery_report(args.output, report)
    except (OSError, ValueError) as exc:
        print(f"coverage audit output failed: {exc}", file=sys.stderr)
        return 4
    print(f"{report['status']}; matrix={report['matrix']['status']}; coverage={report['counts']}")
    return {"verified": 0, "rejected": 1, "unavailable": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
