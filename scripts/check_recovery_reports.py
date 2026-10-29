"""Deterministic diagnostic scenarios; no hardware or physical simulation."""

import argparse

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.recovery_exposure_observer import RecoveryExposureObserver
from embodied_agent.recovery_report import recovery_report, write_recovery_report


class ScenarioSession:
    def __init__(self):
        self.session, self.started = "original", 10.0
        self.status, self.error = "timed_out", "synthetic timeout"
        self.value = None
        self.fail_restart = self.fail_close = False

    def restart(self, factory=None):
        if self.fail_restart:
            raise RuntimeError("synthetic restart failure")
        self.session, self.started = "replacement", 10.5
        self.status, self.error = "idle", None

    def poll(self):
        value, self.value = self.value, None
        return value

    def request(self):
        if self.status not in {"idle", "ready", "error"}:
            return False
        self.status = "pending"
        return True

    def close(self):
        if self.fail_close:
            self.status = "cleanup_failed"
            raise RuntimeError("synthetic cleanup failure")
        self.status = "closed"


def run_scenario(name):
    task = 0.5
    session = ScenarioSession()
    observer = RecoveryExposureObserver(
        session, 0, task_clock=lambda: task, clock=lambda: 10 + task
    )
    if name == "budget_rejected":
        task = 2
    session.fail_restart = name == "restart_failed"
    observer.restart()
    if name in {"restored_pending", "confirmed", "confirmed_cleanup_failed"}:
        task = 0.51
        observer.bind_mapping(session.session, ExposureClockMap(10.51, 0.51), task)
        for task in (0.6,) if name == "restored_pending" else (0.6, 0.8, 1, 1.2, 1.4, 1.6):
            session.status = "ready"
            session.value = StampedPosition((0.64, 0, 0.04), 10 + task - 0.001)
            observer.step(task)
    elif name == "limit_reached":
        session.status, session.error = "failed", "synthetic second failure"
        observer.step(0.6)
        task = 0.7
        observer.restart()
    elif name == "stopped":
        observer.step(0.6, stop=True)
    if name in {"cleanup_failed", "confirmed_cleanup_failed"}:
        session.fail_close = True
    before = recovery_report(observer)
    cleanup_error = None
    try:
        observer.close()
    except RuntimeError as exc:
        cleanup_error = str(exc)
    return {
        "before_close": before,
        "after_close": recovery_report(observer),
        "cleanup_exception": cleanup_error,
    }


def scenarios():
    return {
        "schema_version": 1,
        "test_kind": "deterministic_session_stub",
        "cases": {
            name: run_scenario(name)
            for name in (
                "restored_pending",
                "confirmed",
                "limit_reached",
                "budget_rejected",
                "restart_failed",
                "stopped",
                "cleanup_failed",
                "confirmed_cleanup_failed",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = scenarios()
    write_recovery_report(args.output, report)
    print(f"Wrote {len(report['cases'])} diagnostic scenarios to {args.output}")


if __name__ == "__main__":
    main()
