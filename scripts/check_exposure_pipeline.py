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


def run_case(mode):
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
        exposure = requested + (0.02 if len(calls) % 2 == 0 else 0.06)
        returned = requested + 0.1
        position = (
            (0.62 + 0.025 * (exposure - 100), 0, 0.04) if mode == "moving" else (0.64, 0, 0.04)
        )
        stamped = requested - 0.01 if mode == "cached" else exposure
        calls.append({"requested": requested, "exposure": stamped, "returned": returned})
        set_clock(returned)
        return StampedPosition(position, stamped)

    capture = ObservationCapture(read, clock=clock, require_timestamp=True)
    observer = ExposureObserver(capture, ExposureClockMap(100, 0))
    try:
        for i in range(17):
            task = i / 5
            set_clock(100 + task * (1.04 if mode == "drift" else 1))
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
    args = parser.parse_args()
    cases = [run_case(mode) for mode in ("stationary", "moving", "cached", "drift", "stop")]
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
        "source_sha256": {str(p): digest(p) for p in paths},
        "cases": cases,
    }
    with Path(args.output).open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    assert cases[0]["status"] == "confirmed_in_tray"
    assert all(c["status"] != "confirmed_in_tray" for c in cases[1:])
    assert cases[-1]["status"] == "cancelled"
    print([(c["mode"], c["status"], len(c["calls"])) for c in cases])


if __name__ == "__main__":
    main()
