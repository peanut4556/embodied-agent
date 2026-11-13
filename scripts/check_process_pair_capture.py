"""Spawn isolation smoke check using synthetic readers, not camera hardware."""

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_pair_capture import ProcessPairCapture
from embodied_agent.threaded_pair_capture import CapturedPair
from embodied_agent.two_stage_capture import TwoStageCaptureController
from embodied_agent.two_stage_runtime import TwoStageCaptureRuntime
from scripts.evaluate_sampling_aliases import sources as parent_sources


class FastReader:
    def __call__(self, req):
        stamp = time.monotonic()
        p = StampedPosition((0.64, 0, 0.045), stamp)
        return CapturedPair(req.session_id, p, p)


def fast_factory():
    return FastReader()


class HungReader:
    def __call__(self, req):
        while True:
            time.sleep(0.01)


def hung_factory():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    return HungReader()


def failure_factory():
    raise RuntimeError("synthetic camera initialization failure")


def run(mode, factory):
    b = ProcessPairCapture(factory)
    runtime = None
    try:
        started = time.monotonic()
        ready = b.wait_ready()
        startup_s = time.monotonic() - started
        before = b.result()
        if ready:
            origin = time.monotonic()
            c = TwoStageCaptureController(
                0, ExposureClockMap(origin, 0), "synthetic", position_error_m=(0.001,) * 3
            )
            runtime = TwoStageCaptureRuntime(c, capture_backend=b)
            while c.status == "pending" and time.monotonic() - origin < 4:
                runtime.step()
                time.sleep(0.0005)
            if c.status == "pending":
                runtime.step(stop=True)
        closing = time.monotonic()
        b.cancel()
        cleanup = b.reap(0.25)
        cleanup_s = time.monotonic() - closing
        assert cleanup["workers_stopped"]
        if mode == "hung":
            assert any(w["killed"] for w in cleanup["workers"])
        return {
            "mode": mode,
            "ready": ready,
            "startup_s": startup_s,
            "before_task": before,
            "cleanup_s": cleanup_s,
            "cleanup": cleanup,
            "runtime": runtime.result() if runtime else None,
        }
    finally:
        b.cancel()
        b.reap(0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def sources():
        paths = (
            __file__,
            "src/embodied_agent/process_pair_capture.py",
            "src/embodied_agent/two_stage_runtime.py",
            "src/embodied_agent/threaded_pair_capture.py",
            "src/embodied_agent/two_stage_capture.py",
            "src/embodied_agent/fresh_witness.py",
        )
        return {
            **parent_sources(),
            **{p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        }

    before = sources()
    cases = [
        run("fast", fast_factory),
        run("hung", hung_factory),
        run("startup_failure", failure_factory),
    ]
    assert sources() == before
    path = args.output / "report.json"
    path.write_text(json.dumps({"sources": before, "cases": cases}, indent=2) + "\n")
    summary = {
        "kind": "spawned_synthetic_reader",
        "sources": before,
        "report_path": str(path),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cases": [
            {k: c[k] for k in ("mode", "ready", "startup_s", "cleanup_s", "cleanup")}
            | {
                "status": c["runtime"]["confirmation"]["status"] if c["runtime"] else "not_started",
                "reason": c["runtime"]["confirmation"]["reason"]
                if c["runtime"]
                else "startup_failure",
            }
            for c in cases
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cases"], indent=2))


if __name__ == "__main__":
    main()
