"""Real spawned acquisition with synthetic camera coordinates, no hardware/motion."""

import argparse
import os
import time
from functools import partial

from embodied_agent.camera_process import CameraProcessSession
from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_exposure_observer import ProcessExposureObserver
from embodied_agent.process_post_alert_verification import ProcessPostAlertVerification
from embodied_agent.recovery_report import write_recovery_report


class ScenarioCamera:
    def __init__(self, mode):
        self.mode, self.start = mode, time.monotonic()

    def read(self):
        now = time.monotonic()
        elapsed = now - self.start
        if elapsed >= 2:
            if self.mode == "blocked":
                time.sleep(10)
            elif self.mode == "crashed":
                os._exit(7)
        outside = 1.4 <= elapsed < 2
        return StampedPosition((0.7 if outside else 0.64, 0, 0.04), now)


def run_case(mode):
    session = CameraProcessSession(partial(ScenarioCamera, mode), timeout=0.2)
    try:
        limit = time.monotonic() + 5
        while session.status == "starting" and time.monotonic() < limit:
            session.poll()
            time.sleep(0.005)
        if session.status != "idle":
            raise RuntimeError(f"camera startup failed: {session.status}")
        origin = time.monotonic()
        observer = ProcessExposureObserver(session, 0)
        observer.bind_mapping(
            session.session, ExposureClockMap(origin, 0), time.monotonic() - origin
        )
        limit = time.monotonic() + 1.35
        while observer.status == "pending" and time.monotonic() < limit:
            observer.step(time.monotonic() - origin)
            time.sleep(0.02)
        if observer.status != "confirmed_in_tray":
            raise RuntimeError(f"initial confirmation failed: {observer.result()}")
        original = observer.result()
        adapter = ProcessPostAlertVerification.from_confirmed(observer)
        with adapter:
            observer.close()  # Ownership was transferred; this must not reap the child.
            limit = time.monotonic() + 5
            while adapter.status in {"monitoring", "rechecking"} and time.monotonic() < limit:
                adapter.step(
                    time.monotonic() - origin,
                    stop=mode == "stop" and adapter.status == "rechecking",
                )
                time.sleep(0.02)
            status_before_close = adapter.status
            requests = session.request_id
            adapter.step(time.monotonic() - origin)
            terminal_quiet = session.request_id == requests
        expected = {
            "returned": "reconfirmed",
            "blocked": "invalidated",
            "crashed": "invalidated",
            "stop": "cancelled",
        }[mode]
        result = adapter.result()
        match = (
            status_before_close == expected
            and result["alert"] is not None
            and result["cleanup"]["status"] == "succeeded"
            and session.process is None
            and terminal_quiet
            and original == observer.result()
        )
        return {
            "mode": mode,
            "expected_status": expected,
            "match": match,
            "status_before_close": status_before_close,
            "requests": requests,
            "terminal_quiet": terminal_quiet,
            "process_reaped": session.process is None,
            "exitcode": session.last_exitcode,
            "original_unchanged": original == observer.result(),
            "result": result,
        }
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [run_case(mode) for mode in ("returned", "blocked", "crashed", "stop")]
    report = {
        "schema_version": 1,
        "test_kind": "spawned_synthetic_camera_post_alert",
        "cases": len(rows),
        "mismatches": sum(not row["match"] for row in rows),
        "results": rows,
    }
    write_recovery_report(args.output, report)
    print(f"{len(rows)} spawned camera cases; {report['mismatches']} mismatches")
    return int(report["mismatches"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
