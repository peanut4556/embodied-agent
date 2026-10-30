"""Recompute recorded visual evidence with a fresh confirmation state machine."""

import math

from .exposure_confirmation import ExposureClockMap, ExposureConfirmation
from .observation_capture import StampedPosition

_RESETS = {
    "clock_mapping_bound",
    "awaiting_clock_mapping",
    "camera_session_changed",
    "camera_unavailable",
    "user_stop",
    "observer_closed",
    "camera_cleanup_failed",
    "camera_restart_failed",
    "invalid_recovery_clock",
    "delivery_deadline_reached",
    "restart_limit_reached",
    "insufficient_recovery_budget",
}


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("replay requires finite numeric timestamps/coordinates")
    return value


def replay_report(report):
    """Missing legacy evidence is unavailable, never verified. No camera access.

    Uses the same versioned confirmation algorithm in a fresh instance. This
    checks reproducibility, not independent algorithm correctness or authenticity.
    """
    if "evidence" not in report:
        return {"status": "unavailable", "reason": "legacy_report_without_raw_evidence"}
    try:
        return _replay(report)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f"malformed replay evidence: {exc}") from exc


def _replay(report):
    metadata = report["evidence"]
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
        raise ValueError("unsupported evidence schema")
    if metadata["algorithm"] != "exposure_confirmation_v1":
        raise ValueError("unsupported confirmation algorithm")
    start = _number(metadata["start_time"])
    if report["result"]["deadline"] != start + 3:
        raise ValueError("replay deadline mismatch")
    confirmation, session, last_task = None, None, None
    confirmed = False
    terminal = False
    observations = 0
    restored_sessions = set()
    for row in report["events"]:
        event = row["event"]
        if confirmed or terminal:
            raise ValueError("events recorded after terminal confirmation state")
        if event in ("mapping", "observation", "deadline"):
            task = _number(row["task_time"])
            if task < start or (last_task is not None and task <= last_task):
                raise ValueError("replay task times must be ordered")
            last_task = task
            if event != "deadline" and task >= start + 3:
                raise ValueError("evidence recorded after task deadline")
        if event == "mapping":
            mapping = ExposureClockMap(**row["mapping"])
            host = _number(row["host_time"])
            if mapping.host_origin < _number(row["session_started"]):
                raise ValueError("mapping predates camera session")
            mapping.map(mapping.host_origin, host, task)
            session = row["session"]
            if type(session) is not str or not session:
                raise ValueError("mapping requires a session")
            confirmation = ExposureConfirmation(start, mapping)
        elif event == "observation":
            if confirmation is None or session != row["session"]:
                raise ValueError("observation has no matching session mapping")
            host = _number(row["host_time"])
            point, captured = row["position"], row["captured_at"]
            observation = None
            if point is None:
                if captured is not None:
                    raise ValueError("missing position has an exposure timestamp")
            else:
                if type(point) not in (list, tuple) or len(point) != 3:
                    raise ValueError("replay requires three coordinates")
                observation = StampedPosition(tuple(_number(x) for x in point), _number(captured))
            confirmation.step(task, host, observation)
            calculated = confirmation.result()
            for key, value in calculated.items():
                if key not in row or row[key] != value:
                    raise ValueError(f"replayed observation differs: {key}")
            observations += 1
            if confirmation.last_exposure_time is not None:
                restored_sessions.add(session)
            confirmed = confirmation.status == "confirmed_in_tray"
        elif event == "deadline":
            if task < start + 3:
                raise ValueError("premature deadline event")
            confirmation = None
            terminal = True
        elif event in _RESETS:
            confirmation, session = None, None
            terminal = event in {
                "user_stop",
                "observer_closed",
                "camera_cleanup_failed",
                "camera_restart_failed",
                "invalid_recovery_clock",
                "delivery_deadline_reached",
                "restart_limit_reached",
                "insufficient_recovery_budget",
            }
        else:
            raise ValueError(f"unknown replay event: {event}")
    if confirmed != (report["result"]["status"] == "confirmed_in_tray"):
        raise ValueError("visual confirmation claim is not reproduced")
    for attempt in report["result"]["recovery_attempts"]:
        if (
            attempt["outcome"] == "observation_restored"
            and attempt["new_session"] not in restored_sessions
        ):
            raise ValueError("restored observation claim has no accepted evidence")
    return {
        "status": "verified",
        "observations_replayed": observations,
        "visual_confirmation_reproduced": confirmed,
    }
