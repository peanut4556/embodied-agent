"""Deterministic acquisition/clock fault matrix for process follow-up monitoring."""

import argparse
from collections import Counter
from itertools import pairwise, product

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_post_confirmation import ProcessPostConfirmation
from embodied_agent.recovery_report import write_recovery_report


class ScheduledSession:
    def __init__(self, delay, pattern, drops):
        self.delay, self.pattern, self.drops = delay, pattern, drops
        self.session, self.status, self.error = "camera", "idle", None
        self.task = 1.0
        self.requested_at = None
        self.requests = self.polls = 0
        self.delivered = []

    def request(self):
        if self.status not in {"idle", "ready", "error"}:
            return False
        self.requests += 1
        self.requested_at = self.task
        self.status = "pending"
        return True

    def poll(self):
        self.polls += 1
        if self.status != "pending":
            return None
        elapsed = self.task - self.requested_at
        if elapsed >= 0.25:
            self.status, self.error = "timed_out", "synthetic request timeout"
            return None
        if elapsed < self.delay:
            return None
        self.status = "ready"
        exposure = self.requested_at + self.delay / 2
        outside = self.pattern == "outside" or (self.pattern == "transient" and self.requests == 3)
        point = (0.7 if outside else 0.64, 0, 0.04)
        if self.drops == "all" or (self.drops == "every_third" and self.requests % 3 == 0):
            point = None
        self.delivered.append({"task": self.task, "exposure": exposure, "point": point})
        return None if point is None else StampedPosition(point, 16 + exposure)

    def close(self):
        self.status = "closed"


def reference_alert(deliveries):
    # Independent enumeration of outside-only contiguous exposure windows.
    # Synthetic delivered samples are all finite, fresh and mapped identically.
    for end, last in enumerate(deliveries):
        for begin in range(end):
            window = deliveries[begin : end + 1]
            if last["exposure"] - window[0]["exposure"] < 0.25 - 1e-9:
                continue
            if any(row["point"] is None or row["point"][0] < 0.69 for row in window):
                continue
            if all(0 < b["exposure"] - a["exposure"] <= 0.25 for a, b in pairwise(window)):
                return last["task"]
    return None


def run_case(delay, pattern, drops, fault):
    session = ScheduledSession(delay, pattern, drops)
    offset = 0
    followup = ProcessPostConfirmation(
        session,
        {"status": "confirmed_in_tray"},
        1,
        ExposureClockMap(16, 0),
        clock=lambda: 16 + session.task + offset,
    )
    expected_status, expected_reason = "completed", "followup_window_ended"
    for tick in range(129):
        session.task = 1 + tick / 64
        if session.task >= 1.5:
            if fault == "clock_jump":
                offset = 0.125
                expected_status, expected_reason = "invalidated", "invalid_followup_clock"
            elif fault == "crash":
                session.status, session.error = "failed", "synthetic crash"
                expected_status, expected_reason = "invalidated", "camera_unavailable"
            elif fault == "stop":
                expected_status, expected_reason = "cancelled", "user_stop"
        followup.step(session.task, stop=fault == "stop" and session.task >= 1.5)
        if session.status == "timed_out":
            expected_status, expected_reason = "invalidated", "camera_unavailable"
        if followup.status != "monitoring":
            break
    result = followup.result()
    expected_alert = reference_alert(session.delivered)
    actual_alert = result["alerts"][0]["task_time"] if result["alerts"] else None
    counters = session.requests, session.polls
    followup.step(session.task + 0.25)
    terminal_quiet = counters == (session.requests, session.polls)
    followup.close()
    cleaned = followup.result()["cleanup"]["status"] == "succeeded"
    match = (
        result["status"] == expected_status
        and result["reason"] == expected_reason
        and actual_alert == expected_alert
        and terminal_quiet
        and cleaned
        and result["original_confirmation"]["status"] == "confirmed_in_tray"
    )
    return {
        "delay": delay,
        "pattern": pattern,
        "drops": drops,
        "fault": fault,
        "expected_status": expected_status,
        "actual_status": result["status"],
        "expected_reason": expected_reason,
        "actual_reason": result["reason"],
        "expected_alert_time": expected_alert,
        "actual_alert_time": actual_alert,
        "requests": session.requests,
        "delivered_frames": len(session.delivered),
        "terminal_quiet": terminal_quiet,
        "cleaned": cleaned,
        "match": match,
        "coverage": result["coverage"],
        **({"deliveries": session.delivered, "result": result} if not match else {}),
    }


def evaluate():
    rows = [
        run_case(*args)
        for args in product(
            (0.03125, 0.125, 0.28125),
            ("inside", "outside", "transient"),
            ("none", "every_third", "all"),
            ("none", "clock_jump", "crash", "stop"),
        )
    ]
    return {
        "schema_version": 1,
        "test_kind": "deterministic_session_fault_matrix",
        "cases": len(rows),
        "mismatches": sum(not row["match"] for row in rows),
        "status_counts": dict(Counter(row["actual_status"] for row in rows)),
        "alert_cases": sum(row["actual_alert_time"] is not None for row in rows),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate()
    write_recovery_report(args.output, report)
    print(f"{report['cases']} scenarios; {report['mismatches']} mismatches")
    return int(report["mismatches"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
