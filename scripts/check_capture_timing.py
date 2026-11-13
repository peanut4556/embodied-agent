"""Trace real spawned-process timing with synthetic causal delay injections."""

import argparse
import functools
import hashlib
import json
import time
from pathlib import Path

from embodied_agent.capture_timing import analyze_capture_timing
from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_pair_capture import ProcessPairCapture
from embodied_agent.threaded_pair_capture import CapturedPair
from embodied_agent.two_stage_capture import TwoStageCaptureController
from embodied_agent.two_stage_runtime import TwoStageCaptureRuntime
from scripts.evaluate_sampling_aliases import sources as parent_sources


class DelayedReader:
    def __init__(self, before, after):
        self.before, self.after = before, after

    def __call__(self, req):
        if self.before:
            time.sleep(self.before)
        p = StampedPosition((0.64, 0, 0.045), time.monotonic())
        if self.after:
            time.sleep(self.after)
        return CapturedPair(req.session_id, p, p)


class DelayedDispatch(ProcessPairCapture):
    def submit(self, request):
        time.sleep(0.005)
        return super().submit(request)


def run(mode):
    factory = functools.partial(
        DelayedReader,
        0.005 if mode == "reader_delay" else 0,
        0.005 if mode == "return_delay" else 0,
    )
    backend_type = DelayedDispatch if mode == "dispatch_delay" else ProcessPairCapture
    b = backend_type(factory, trace_timing=True)
    try:
        assert b.wait_ready()
        origin = time.monotonic()
        c = TwoStageCaptureController(
            0, ExposureClockMap(origin, 0), "timing-audit", position_error_m=(0.001,) * 3
        )
        runtime = TwoStageCaptureRuntime(c, capture_backend=b)
        while c.status == "pending" and time.monotonic() - origin < 4:
            runtime.step()
            time.sleep(0.0005)
        if c.status == "pending":
            runtime.step(stop=True)
        cleanup = runtime.close(wait_s=0.25)
        assert cleanup["workers_stopped"]
        rows = []
        requests = {r["request_id"]: r for r in c.requests}
        for trace in b.result()["timing_trace"]:
            req = requests[trace["request_id"]]
            # Only candidate requests have the fixed 2 ms schedule contract.
            if req["stage"] != "candidate":
                continue
            target = origin + c.candidate.schedule[req["request_id"] - 1]
            rows.append(analyze_capture_timing(trace, scheduled_host_at=target))
        measured = [r for r in rows if r["status"] == "measured"]
        # Check injected latency appears in the corresponding segment, not merely
        # that some end-to-end timeout occurred. Do not assert baseline success.
        expected = {
            "dispatch_delay": "caller_dispatch",
            "reader_delay": "reader_to_exposure",
            "return_delay": "exposure_to_reader_return",
        }.get(mode)
        assert measured
        if expected:
            assert all(r["segments_s"][expected] >= 0.0045 for r in measured)
        if mode in ("dispatch_delay", "reader_delay"):
            assert c.status == "unconfirmed"
            assert not measured[0]["exposure_window_met"]
        return {"mode": mode, "runtime": runtime.result(), "analysis": rows}
    finally:
        b.cancel()
        b.reap(0.25)


def sources():
    paths = (
        "scripts/check_capture_timing.py",
        "src/embodied_agent/capture_timing.py",
        "src/embodied_agent/process_pair_capture.py",
        "src/embodied_agent/two_stage_runtime.py",
        "src/embodied_agent/threaded_pair_capture.py",
        "src/embodied_agent/two_stage_capture.py",
        "src/embodied_agent/fresh_witness.py",
    )
    return parent_sources() | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = sources()
    cases = [run(mode) for mode in ("baseline", "dispatch_delay", "reader_delay", "return_delay")]
    assert sources() == before
    report = args.output / "report.json"
    report.write_text(json.dumps({"sources": before, "cases": cases}, indent=2) + "\n")
    summary = {
        "kind": "synthetic_reader_real_process_timing",
        "sources": before,
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "cases": [],
    }
    for case in cases:
        confirmation = case["runtime"]["confirmation"]
        measured = [r for r in case["analysis"] if r["status"] == "measured"]
        summary["cases"].append(
            {
                "mode": case["mode"],
                "status": confirmation["status"],
                "reason": confirmation["reason"],
                "requests": len(confirmation["requests"]),
                "candidate_traces": len(case["analysis"]),
                "measured": len(measured),
                "missed_exposure_windows": sum(not r["exposure_window_met"] for r in measured),
                "max_segments_ms": {
                    k: 1000 * max(r["segments_s"][k] for r in measured)
                    for k in measured[0]["segments_s"]
                },
                "cleanup_verified": case["runtime"]["capture"]["workers_stopped"],
            }
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cases"], indent=2))


if __name__ == "__main__":
    main()
