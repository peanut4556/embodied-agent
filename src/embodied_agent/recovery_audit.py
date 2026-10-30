"""Offline consistency checks, not authentication or replay of visual evidence."""

import json
import math
from collections import Counter
from pathlib import Path

from .recovery_replay import replay_report


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return type(value) in {int, float} and math.isfinite(value)


def validate_report(report):
    """Reject contradictory v1 snapshots; return a summary from checked fields."""
    _require(type(report) is dict, "report must be an object")
    _require(
        type(report.get("schema_version")) is int and report["schema_version"] == 1,
        "unsupported report schema",
    )
    _require(report.get("scope") == "experimental_visual_release_confirmation", "unknown scope")
    for key in ("result", "diagnosis", "cleanup", "camera", "policy"):
        _require(type(report.get(key)) is dict, f"missing object: {key}")
    result, diagnosis, cleanup = report["result"], report["diagnosis"], report["cleanup"]
    _require(
        type(result.get("schema_version")) is int and result["schema_version"] == 2,
        "unsupported result schema",
    )
    status = result.get("status")
    _require(
        status
        in (
            "pending",
            "cancelled",
            "unconfirmed",
            "confirmed_in_tray",
            "outside_confirmation_region",
        ),
        "invalid confirmation status",
    )
    _require(type(result.get("reason")) is str and bool(result["reason"]), "missing reason")
    _require(_finite(result.get("deadline")), "invalid deadline")
    _require(type(result.get("mapping_bound")) is bool, "invalid mapping flag")
    for key in ("restart_count", "max_restarts"):
        _require(type(result.get(key)) is int and result[key] >= 0, f"invalid {key}")
    attempts = result.get("recovery_attempts")
    _require(type(attempts) is list, "missing attempt list")
    _require(
        len(attempts) == result["restart_count"] <= result["max_restarts"],
        "restart count/limit mismatch",
    )
    restored = []
    previous = None
    for index, attempt in enumerate(attempts, 1):
        _require(type(attempt) is dict, "invalid attempt")
        _require(
            type(attempt.get("attempt")) is int and attempt["attempt"] == index,
            "attempt numbers must be contiguous",
        )
        started = attempt.get("task_time")
        _require(_finite(started) and started < result["deadline"], "invalid attempt time")
        _require(previous is None or started > previous, "unordered attempts")
        previous = started
        _require(attempt.get("fault") in ("failed", "timed_out"), "invalid recovery fault")
        _require(
            type(attempt.get("old_session")) is str and bool(attempt["old_session"]),
            "missing old session",
        )
        outcome = attempt.get("outcome")
        _require(
            outcome
            in (
                "restarting",
                "restart_failed",
                "awaiting_clock_mapping",
                "invalid_task_clock",
                "deadline_exceeded",
                "observation_restored",
                "cancelled",
                "unconfirmed",
                "camera_failed_again",
                "observer_closed",
                "cleanup_failed",
                "delivery_deadline_reached",
                "restart_limit_reached",
                "insufficient_recovery_budget",
                "camera_cleanup_failed",
            ),
            "unknown attempt outcome",
        )
        if "completed_at" in attempt:
            _require(
                _finite(attempt["completed_at"]) and attempt["completed_at"] >= started,
                "invalid restart completion time",
            )
            previous = attempt["completed_at"]
        if outcome == "observation_restored":
            _require(
                type(attempt.get("new_session")) is str
                and bool(attempt["new_session"])
                and attempt["new_session"] != attempt["old_session"],
                "restored observation requires a new session",
            )
            _require(
                "completed_at" in attempt and attempt["completed_at"] < result["deadline"],
                "restored observation requires restart within deadline",
            )
            restored.append(index)
    expected = {
        "observation_restored": bool(restored),
        "visual_confirmation_succeeded": status == "confirmed_in_tray",
        "terminal": status != "pending",
    }
    cleanup_status = cleanup.get("status")
    _require(cleanup_status in ("not_requested", "succeeded", "failed"), "invalid cleanup status")
    _require(
        (type(cleanup.get("error")) is str)
        if cleanup_status == "failed"
        else cleanup.get("error") is None,
        "cleanup error/status mismatch",
    )
    if cleanup_status == "succeeded":
        _require(
            report["camera"].get("status") == "closed", "successful cleanup requires closed camera"
        )
    expected["cleanup_required"] = cleanup_status != "succeeded"
    for key, value in expected.items():
        _require(
            type(diagnosis.get(key)) is bool and diagnosis[key] == value,
            f"contradictory diagnosis: {key}",
        )
    ids = diagnosis.get("restored_attempts")
    _require(
        type(ids) is list and all(type(x) is int for x in ids) and ids == restored,
        "restored attempt list mismatch",
    )
    _require(diagnosis.get("reason_code") == result["reason"], "reason code mismatch")
    _require(
        _finite(report["policy"].get("min_remaining")) and report["policy"]["min_remaining"] >= 1,
        "invalid recovery reserve",
    )
    _require(
        type(report.get("events")) is list and all(type(row) is dict for row in report["events"]),
        "invalid event list",
    )
    return {
        **expected,
        "status": status,
        "reason": result["reason"],
        "cleanup_status": cleanup_status,
        "evidence_replay": replay_report(report),
        "restart_count": len(attempts),
    }


def _pairs(items):
    value = {}
    for key, item in items:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _bad_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _number(value):
    number = float(value)
    _require(math.isfinite(number), "non-finite JSON number")
    return number


def audit_files(paths):
    """Audit standalone snapshots or scenario bundles; count only after_close.

    Invalid scenarios are excluded from totals. Duplicate resolved input paths
    are rejected rather than counted twice. This cannot identify copied runs.
    """
    accepted, rejected, seen = [], [], set()
    for path in paths:
        source = str(Path(path).resolve())
        if source in seen:
            rejected.append({"source": source, "error": "duplicate input path"})
            continue
        seen.add(source)
        try:
            report = json.loads(
                Path(path).read_text(encoding="utf-8"),
                object_pairs_hook=_pairs,
                parse_constant=_bad_constant,
                parse_float=_number,
            )
            _require(type(report) is dict, "input must be an object")
            if "cases" in report:
                _require(
                    type(report.get("schema_version")) is int
                    and report["schema_version"] == 1
                    and report.get("test_kind") == "deterministic_session_stub",
                    "unsupported scenario bundle",
                )
                _require(type(report["cases"]) is dict and bool(report["cases"]), "empty cases")
                entries = report["cases"].items()
            else:
                entries = [(None, {"after_close": report})]
            for name, entry in entries:
                try:
                    _require(type(entry) is dict, "invalid case")
                    if name is not None:
                        before = validate_report(entry.get("before_close"))
                    summary = validate_report(entry.get("after_close"))
                    if name is not None:
                        expected_status = (
                            "cancelled" if before["status"] == "pending" else before["status"]
                        )
                        _require(
                            summary["status"] == expected_status,
                            "close changed confirmation inconsistently",
                        )
                        for key in ("deadline", "max_restarts", "restart_count", "session"):
                            _require(
                                entry["before_close"]["result"].get(key)
                                == entry["after_close"]["result"].get(key),
                                f"close changed {key}",
                            )
                        _require(
                            summary["cleanup_status"] != "not_requested",
                            "after_close has no cleanup result",
                        )
                        _require(
                            entry.get("cleanup_exception")
                            == entry["after_close"]["cleanup"].get("error"),
                            "cleanup exception mismatch",
                        )
                    accepted.append({"source": source, "case": name, **summary})
                except ValueError as exc:
                    rejected.append({"source": source, "case": name, "error": str(exc)})
        except (OSError, UnicodeError, ValueError) as exc:
            rejected.append({"source": source, "error": str(exc)})
    return {
        "schema_version": 1,
        "audit_scope": "state_consistency_and_available_evidence_replay",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "counts": {
            key: sum(row[key] for row in accepted)
            for key in (
                "observation_restored",
                "visual_confirmation_succeeded",
                "terminal",
                "cleanup_required",
                "restart_count",
            )
        },
        "status_counts": dict(Counter(row["status"] for row in accepted)),
        "cleanup_counts": dict(Counter(row["cleanup_status"] for row in accepted)),
        "replay_counts": dict(Counter(row["evidence_replay"]["status"] for row in accepted)),
        "accepted": accepted,
        "rejected": rejected,
    }
