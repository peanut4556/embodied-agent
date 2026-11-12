"""Real-thread/real-monotonic-clock smoke test with synthetic camera readers."""

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.threaded_pair_capture import CapturedPair
from embodied_agent.two_stage_capture import TwoStageCaptureController
from embodied_agent.two_stage_runtime import TwoStageCaptureRuntime
from scripts.evaluate_sampling_aliases import sources as parent_sources


def run(mode):
    release = threading.Event()
    origin = time.monotonic()

    def read(request, cancel):
        captured = time.monotonic()
        if mode == "cooperative_stall":
            cancel.wait(5)
        elif mode == "uncooperative_stall":
            release.wait(5)
        else:
            cancel.wait(0.002 if request.stage == "candidate" else 0.001)
        p = StampedPosition((0.64, 0, 0.045), captured)
        return CapturedPair(request.session_id, p, p)

    c = TwoStageCaptureController(
        0, ExposureClockMap(origin, 0), "synthetic", position_error_m=(0.001,) * 3
    )
    runtime = TwoStageCaptureRuntime(c, read)
    peak_step = 0
    try:
        while c.status == "pending" and time.monotonic() - origin < 4:
            started = time.monotonic()
            runtime.step()
            peak_step = max(peak_step, time.monotonic() - started)
            time.sleep(0.0005)
        if c.status == "pending":
            runtime.step(stop=True)
        before_release = runtime.close(wait_s=0.02)
    finally:
        release.set()
        after_release = runtime.close(wait_s=0.25)
    assert after_release["workers_stopped"]
    if mode == "uncooperative_stall":
        assert not before_release["workers_stopped"]
    return {
        "mode": mode,
        "wall_duration_s": time.monotonic() - origin,
        "max_observed_step_s": peak_step,
        "before_release": before_release,
        "after_release": after_release,
        "runtime": runtime.result(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def sources():
        paths = (
            __file__,
            "src/embodied_agent/threaded_pair_capture.py",
            "src/embodied_agent/two_stage_runtime.py",
            "src/embodied_agent/two_stage_capture.py",
            "src/embodied_agent/fresh_witness.py",
        )
        return {
            **parent_sources(),
            **{p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        }

    before = sources()
    cases = [run(mode) for mode in ("fast", "cooperative_stall", "uncooperative_stall")]
    assert before == sources()
    path = args.output / "report.json"
    path.write_text(
        json.dumps(
            {"kind": "real_threads_synthetic_reader", "sources": before, "cases": cases}, indent=2
        )
        + "\n"
    )
    summary = {
        "kind": "real_threads_synthetic_reader",
        "sources": before,
        "report_path": str(path),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cases": [
            {
                "mode": c["mode"],
                "wall_duration_s": c["wall_duration_s"],
                "max_observed_step_s": c["max_observed_step_s"],
                "status": c["runtime"]["confirmation"]["status"],
                "reason": c["runtime"]["confirmation"]["reason"],
                "requests": len(c["runtime"]["confirmation"]["requests"]),
                "before_release": c["before_release"],
                "after_release": c["after_release"],
            }
            for c in cases
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cases"], indent=2))


if __name__ == "__main__":
    main()
