"""Deterministic camera stub with real worker threads; not a physical benchmark."""

import argparse
import json
import threading
import time
from pathlib import Path

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.exposure_observer import ExposureObserver
from embodied_agent.memory_policy import digest
from embodied_agent.observation_capture import ObservationCapture, StampedPosition

BASELINE = ("stationary", "moving", "cached", "drift", "stop")
STRESS = (
    "jitter",
    "moving_jitter",
    "exposure_gaps",
    "early_drop",
    "periodic_drop",
    "all_drop",
    "clock_jump",
    "clock_recovery",
    "late_poll",
)


def run_case(mode):
    if mode not in BASELINE + STRESS:
        raise ValueError("unknown camera schedule")
    now, calls = [100.0], []
    lock = threading.Lock()

    def clock():
        with lock:
            return now[0]

    def set_clock(value):
        with lock:
            now[0] = value

    def read(cancel):
        requested = clock()
        index = len(calls)
        offset = 0.02 if index % 2 == 0 else 0.06
        duration = 0.1
        if mode in {"jitter", "moving_jitter"}:
            offset = (0.01, 0.045, 0.02, 0.055)[index % 4]
            duration = (0.08, 0.18, 0.12, 0.16)[index % 4]
        elif mode == "exposure_gaps":
            offset = 0.01 if index % 2 == 0 else 0.09
        exposure = requested + offset
        returned = requested + duration
        missing = (
            mode == "all_drop"
            or (mode == "early_drop" and index < 4)
            or (mode == "periodic_drop" and index % 3 == 2)
        )
        position = (
            (0.62 + 0.025 * (exposure - 100), 0, 0.04)
            if mode in {"moving", "moving_jitter"}
            else (0.64, 0, 0.04)
        )
        stamped = requested - 0.01 if mode == "cached" else exposure
        calls.append(
            {"requested": requested, "exposure": stamped, "returned": returned, "missing": missing}
        )
        set_clock(returned)
        return None if missing else StampedPosition(position, stamped)

    capture = ObservationCapture(read, clock=clock, require_timestamp=True)
    observer = ExposureObserver(capture, ExposureClockMap(100, 0))
    try:
        schedule = [0, 0.2, 0.4, 3.1] if mode == "late_poll" else [i / 5 for i in range(17)]
        for i, task in enumerate(schedule):
            offset = (
                0.05
                if ((mode == "clock_jump" and i >= 3) or (mode == "clock_recovery" and i in (3, 4)))
                else 0
            )
            set_clock(100 + task * (1.04 if mode == "drift" else 1) + offset)
            observer.step(task, [{"event": "intentional_release"}], stop=mode == "stop" and i == 3)
            # Ensure each scheduled return completes before advancing the stub clock.
            limit = time.monotonic() + 2
            while True:
                with capture.lock:
                    done = capture.job is None or capture.job["done"]
                if done:
                    break
                if time.monotonic() > limit:
                    raise RuntimeError("camera stub did not return")
                time.sleep(0.001)
        return {
            "mode": mode,
            "status": observer.status,
            "calls": calls,
            "observations": observer.rows,
        }
    finally:
        capture.cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", choices=("baseline", "stress"), default="baseline")
    args = parser.parse_args()
    cases = [run_case(mode) for mode in BASELINE + (STRESS if args.suite == "stress" else ())]
    paths = [Path(__file__)] + [
        Path("src/embodied_agent") / p
        for p in (
            "exposure_observer.py",
            "exposure_confirmation.py",
            "observation_capture.py",
            "release_confirmation.py",
        )
    ]
    report = {
        "scope": "Threaded camera stub with controlled clocks; no hardware or physics.",
        "suite": args.suite,
        "source_sha256": {str(p): digest(p) for p in paths},
        "cases": cases,
    }
    with Path(args.output).open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert cases[0]["status"] == "confirmed_in_tray"
    assert all(c["status"] != "confirmed_in_tray" for c in cases[1:5])
    assert cases[4]["status"] == "cancelled"
    if args.suite == "stress":
        expected_success = {"jitter", "early_drop", "clock_recovery"}
        for case in cases[5:]:
            assert (case["status"] == "confirmed_in_tray") == (case["mode"] in expected_success), (
                case["mode"]
            )
    print([(c["mode"], c["status"], len(c["calls"])) for c in cases])


if __name__ == "__main__":
    main()
