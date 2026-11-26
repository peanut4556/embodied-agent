"""Bounded offline integrity checks for recalibration lifecycle evidence."""

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

from .budget_checkpoint import digest, fields
from .recalibration_stage_results import check_stage_results
from .recovery_json import parse_recovery_json

LIMIT = 1_048_576
STAGES = (
    ("diagnose-invalid", 1),
    ("preview-invalid", 3),
    ("diagnose-fixed", 0),
    ("preview-fixed", 0),
    ("reject-stale-preview", 3),
    ("preview-current", 0),
    ("apply-current", 0),
    ("status-applied", 0),
    ("reject-repeat-apply", 3),
    ("status-after-drift", 0),
    ("diagnose-reused-batches", 1),
)
ARTIFACTS = {
    "budget.json",
    "plan.json",
    "invalid-plan.json",
    "initial.json",
    "concurrent-change.json",
    "applied.json",
    "drifted.json",
}


def _read(path):
    with path.open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError("evidence file exceeds size limit")
    return raw


def _relative(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid evidence path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("unsafe evidence path")
    return path.parts


def _hash_files(root, manifest):
    rows = []
    for name, expected in sorted(manifest.items()):
        path = root
        for part in _relative(name):
            path = path / part
            if path.is_symlink():
                raise ValueError("symlink in evidence path")
        if not path.exists():
            rows.append({"path": name, "status": "missing"})
            continue
        if not path.is_file():
            raise ValueError("evidence entry is not a regular file")
        actual = hashlib.sha256(_read(path)).hexdigest()
        rows.append({"path": name, "status": "matched" if actual == expected else "changed"})
    return rows


def verify_evidence(directory, *, source_root=None, check_results=False):
    if type(check_results) is not bool:
        raise TypeError("check_results must be boolean")
    root = Path(directory).resolve()
    report_path = root / "report.json"
    if report_path.is_symlink():
        raise ValueError("report symlink rejected")
    raw = _read(report_path)
    report = parse_recovery_json(raw.decode("utf-8"))
    fields(report, "passed environment python stages source_sha256 artifact_sha256")
    if type(report["passed"]) is not bool or report["environment"] != "synthetic-only":
        raise ValueError("invalid evidence header")
    if not isinstance(report["python"], str):
        raise TypeError("invalid python description")
    stages = report["stages"]
    if not isinstance(stages, list) or len(stages) != len(STAGES):
        raise ValueError("incomplete lifecycle stages")
    stage_issues = []
    for row, (name, code) in zip(stages, STAGES, strict=True):
        fields(row, "name exit_code expected_exit_code stdout stderr")
        if (
            row["name"] != name
            or type(row["exit_code"]) is not int
            or type(row["expected_exit_code"]) is not int
            or not isinstance(row["stdout"], str)
            or not isinstance(row["stderr"], str)
        ):
            raise ValueError("invalid lifecycle stage")
        if row["exit_code"] != code or row["expected_exit_code"] != code:
            stage_issues.append(name)
    for key in ("artifact_sha256", "source_sha256"):
        manifest = report[key]
        if not isinstance(manifest, dict) or not 1 <= len(manifest) <= 256:
            raise ValueError("invalid evidence manifest")
        for name, checksum in manifest.items():
            _relative(name)
            digest(checksum)
    if set(report["artifact_sha256"]) != ARTIFACTS:
        raise ValueError("incomplete artifact manifest")
    for name in report["source_sha256"]:
        parts = _relative(name)
        if not (
            name == "scripts/validate_recalibration_lifecycle.py"
            or (
                len(parts) == 3
                and parts[:2] == ("src", "embodied_agent")
                and parts[2].endswith(".py")
            )
        ):
            raise ValueError("unexpected source path")
    if "scripts/validate_recalibration_lifecycle.py" not in report["source_sha256"]:
        raise ValueError("missing lifecycle source")
    artifacts = _hash_files(root, report["artifact_sha256"])
    sources = (
        _hash_files(Path(source_root).resolve(), report["source_sha256"])
        if source_root is not None
        else []
    )
    intact = all(r["status"] == "matched" for r in artifacts + sources)
    result_issues = check_stage_results(stages) if check_results else []
    return {
        "verified": report["passed"] and not stage_issues and intact and not result_issues,
        "stage_results_checked": check_results,
        "stage_result_issues": result_issues,
        "scope": (
            "recorded_integrity_exit_codes_and_scenario_assertions"
            if check_results
            else "recorded_file_integrity_and_stage_exit_codes"
        ),
        "authenticated": False,
        "source_checked": source_root is not None,
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "stage_issues": stage_issues,
        "artifacts": artifacts,
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--check-stage-results", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_evidence(
            args.directory, source_root=args.source_root, check_results=args.check_stage_results
        )
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"ok": False, "cause_type": type(exc).__name__}), file=sys.stderr)
        return 3
    print(json.dumps({"ok": True, "result": result}))
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
