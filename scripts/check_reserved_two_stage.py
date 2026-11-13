"""Real-process end-to-end confirmation with opt-in advance reservations."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_pair_capture import ProcessPairCapture
from embodied_agent.reserved_two_stage import ReservedTwoStageController
from embodied_agent.scheduled_pair_capture import ScheduledReaderFactory
from embodied_agent.threaded_pair_capture import CapturedPair
from embodied_agent.two_stage_capture import TwoStageCaptureController
from embodied_agent.two_stage_runtime import TwoStageCaptureRuntime
from scripts.check_capture_timing import DelayedDispatch
from scripts.check_capture_timing import sources as parent_sources
from scripts.check_process_pair_capture import fast_factory


class WitnessMotionReader:
    def __call__(self, req):
        p = StampedPosition((0.642 if req.stage == "witness" else 0.64, 0, 0.045), time.monotonic())
        return CapturedPair(req.session_id, p, p)


def motion_factory():
    return WitnessMotionReader()


def run(profile, mode, repetition):
    factory = motion_factory if mode == "witness_motion" else fast_factory
    if profile == "reserved":
        factory = ScheduledReaderFactory(factory)
    backend_type = DelayedDispatch if mode == "dispatch_delay" else ProcessPairCapture
    backend = backend_type(factory, trace_timing=True)
    try:
        assert backend.wait_ready()
        origin = time.monotonic()
        cls = ReservedTwoStageController if profile == "reserved" else TwoStageCaptureController
        c = cls(
            0.02 if profile == "reserved" else 0,
            ExposureClockMap(origin, 0),
            "synthetic-full-flow",
            position_error_m=(0.001,) * 3,
        )
        runtime = TwoStageCaptureRuntime(c, capture_backend=backend)
        while c.status == "pending" and time.monotonic() - origin < 4:
            runtime.step()
            if mode == "stop_reserved" and c.requests:
                runtime.step(stop=True)
            time.sleep(0.0005)
        if c.status == "pending":
            runtime.step(stop=True)
        cleanup = runtime.close(wait_s=0.25)
        assert cleanup["workers_stopped"]
        if c.status == "confirmed_recent_evidence":
            assert len(c.requests) == 36 and len(c.gate.rows) == 5
        if mode == "stop_reserved":
            assert c.status == "cancelled" and len(c.requests) == 1
        if mode == "witness_motion":
            # Earlier timing failures remain valid and are reported, not hidden.
            assert c.status != "confirmed_recent_evidence"
        return {
            "profile": profile,
            "mode": mode,
            "repetition": repetition,
            "runtime": runtime.result(),
        }
    finally:
        backend.cancel()
        backend.reap(0.25)


def sources():
    return parent_sources() | {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (
            "scripts/check_reserved_two_stage.py",
            "scripts/check_process_pair_capture.py",
            "src/embodied_agent/reserved_two_stage.py",
            "src/embodied_agent/scheduled_pair_capture.py",
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = sources()
    cases = []
    for repetition in range(2):
        for mode in ("baseline", "dispatch_delay"):
            profiles = ("immediate", "reserved") if repetition == 0 else ("reserved", "immediate")
            for profile in profiles:
                cases.append(run(profile, mode, repetition))
    cases.extend(run("reserved", mode, 0) for mode in ("witness_motion", "stop_reserved"))
    assert before == sources()
    report = args.output / "report.json"
    report.write_text(json.dumps({"sources": before, "cases": cases}, indent=2) + "\n")
    summary = {
        "kind": "synthetic_reserved_two_stage_full_confirmation",
        "sources": before,
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "cases": [],
    }
    for case in cases:
        c = case["runtime"]["confirmation"]
        summary["cases"].append(
            {
                "profile": case["profile"],
                "mode": case["mode"],
                "repetition": case["repetition"],
                "status": c["status"],
                "reason": c["reason"],
                "requests": len(c["requests"]),
                "candidate_status": c["candidate"]["status"],
                "witness_observations": len(c["witness"]["observations"]) if c["witness"] else 0,
                "driver_errors": case["runtime"]["driver_errors"],
                "workers_stopped": case["runtime"]["capture"]["workers_stopped"],
            }
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cases"], indent=2))


if __name__ == "__main__":
    main()
