"""Deterministic synthetic scheduling experiment; no camera or physics claims."""

import argparse
import hashlib
import json
from pathlib import Path

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.pipelined_challenge import PipelinedChallengeController

SOURCES = [
    __file__,
    *[
        f"src/embodied_agent/{name}.py"
        for name in (
            "pipelined_challenge",
            "serial_challenge",
            "delivered_challenge",
            "confirmation_challenge",
            "exposure_confirmation",
            "observation_capture",
            "settled_reference",
            "reference_confirmation",
            "temporal_reference",
            "cross_view",
            "release_confirmation",
        )
    ],
]


def hashes():
    return {str(Path(p)): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}


def run(delay, capacity, motion):
    controller = PipelinedChallengeController(
        0,
        ExposureClockMap(0, 0),
        "synthetic",
        position_error_m=(0.001,) * 3,
        delivery_budget_s=0.1,
        capacity=capacity,
    )
    jobs = {}
    for ms in range(3101):
        now = ms / 1000
        if ms in jobs:
            request = jobs.pop(ms)
            position = (0.642 if motion and request.task_time >= 2.05 else 0.64, 0, 0.045)
            frame = StampedPosition(position, request.host_time)
            controller.complete(request.request_id, now, now, frame, frame, session_id="synthetic")
        request = controller.poll(now, now)
        if request:
            jobs[ms + delay] = request
        if controller.status != "pending":
            break
    result = controller.result()
    expected = not motion and delay <= 100 and capacity * 50 >= delay
    assert (result["status"] == "confirmed_in_tray") == expected
    assert result["status"] != "pending"
    assert result["peak_inflight"] <= capacity
    assert result["request_count"] <= 31
    assert all(r["task_time"] <= 3 for r in result["requests"])
    return {"delay_ms": delay, "capacity": capacity, "motion": motion, "result": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = hashes()
    cases = [run(d, c, m) for m in (False, True) for d in (20, 50, 80, 100, 120) for c in (1, 2, 3)]
    assert hashes() == before
    report = {
        "schema_version": 1,
        "kind": "synthetic_virtual_time",
        "sources": before,
        "cases": cases,
    }
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        "schema_version": 1,
        "kind": report["kind"],
        "sources": before,
        "report_path": str(path),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cases": [
            dict(
                delay_ms=c["delay_ms"],
                capacity=c["capacity"],
                motion=c["motion"],
                **{
                    k: c["result"][k]
                    for k in ("status", "reason", "request_count", "peak_inflight")
                },
            )
            for c in cases
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Validated {len(cases)} synthetic cases: {path}")


if __name__ == "__main__":
    main()
