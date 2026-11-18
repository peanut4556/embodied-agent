"""Fixed-denominator independent exposure probes; never task confirmation/retry."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from embodied_agent.process_pair_capture import ProcessPairCapture
from embodied_agent.scheduled_pair_capture import ScheduledPairRequest, ScheduledReaderFactory
from scripts.check_evidence_checkpoint import digest
from scripts.check_process_pair_capture import fast_factory


def summarize(rows, trace, planned):
    ids = [row["request_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(range(1, planned + 1)):
        raise ValueError("exactly one outcome per planned request required")
    timing_ids = [row["request_id"] for row in trace]
    if len(timing_ids) != len(set(timing_ids)) or not set(timing_ids) <= set(ids):
        raise ValueError("unique matching trace ids required")
    waits = []
    for row in trace:
        wait = row.get("reservation_wait")
        if wait is not None:
            if wait["request_id"] != row["request_id"]:
                raise ValueError("wait metrics identity mismatch")
            waits.append(wait)
    counts = dict(Counter(row["outcome"] for row in rows))
    complete = len(waits) == planned and all(
        row["outcome"] in ("delivered", "reader_error") for row in rows
    )
    return {
        "planned": planned,
        "outcomes": counts,
        "measured": len(waits),
        "measurement_complete": complete,
        "wait_cpu_s": sum(w["wait_cpu_s"] for w in waits),
        "mean_wait_cpu_s": sum(w["wait_cpu_s"] for w in waits) / planned if complete else None,
        "max_target_lateness_s": max((w["target_lateness_s"] for w in waits), default=None),
    }


def run(margin, *, wait_cpu_budget_s=None):
    backend = ProcessPairCapture(
        ScheduledReaderFactory(fast_factory, margin, True, wait_cpu_budget_s),
        capacity=1,
        supervised=True,
        trace_timing=True,
        runtime_limit_s=5,
    )
    rows = []
    try:
        ready = backend.wait_ready()
        stopped = not ready
        for request_id in range(1, 37):
            if stopped:
                rows.append({"request_id": request_id, "outcome": "not_attempted"})
                continue
            now = time.monotonic()
            req = ScheduledPairRequest(
                request_id, "independent-probe", now, now + 0.02, now + 0.022
            )
            if not backend.submit(req):
                rows.append({"request_id": request_id, "outcome": "submit_refused"})
                stopped = True
                continue
            deadline = time.monotonic() + 0.2
            packets = []
            while not packets and time.monotonic() < deadline:
                packets = backend.drain()
                time.sleep(0.0005)
            if not packets:
                rows.append({"request_id": request_id, "outcome": "timeout"})
                stopped = True
                continue
            identity, pair, error = packets[0]
            assert identity == request_id
            rows.append(
                {
                    "request_id": request_id,
                    "outcome": "reader_error" if error else "delivered",
                    "error": error,
                    "target_host_time": req.target_host_time,
                    "latest_host_time": req.latest_host_time,
                    "exposures": None
                    if pair is None
                    else [pair.primary.captured_at, pair.secondary.captured_at],
                }
            )
            # A completed error ends THIS probe; next id is a new independent
            # window, not a retry or continuation of a failed confirmation.
    finally:
        backend.cancel()
        cleanup = backend.reap(0.25)
    summary = summarize(rows, cleanup["timing_trace"], 36)
    return {"spin_window_s": margin, "rows": rows, "capture": cleanup, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_fixed_wait_probe.py",
        "scripts/check_process_pair_capture.py",
        "src/embodied_agent/process_pair_capture.py",
        "src/embodied_agent/capture_supervisor.py",
        "src/embodied_agent/scheduled_pair_capture.py",
    ]
    sources = {p: digest(p) for p in paths}
    cases = [run(margin) for margin in (0, 0.0015, 0.0015, 0)]
    assert sources == {p: digest(p) for p in paths}
    report = out / "report.json"
    report.write_text(json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n")
    summary = {
        "sources": sources,
        "report_path": str(report),
        "report_sha256": digest(report),
        "cases": [
            {
                "spin_window_s": c["spin_window_s"],
                **c["summary"],
                "workers_stopped": c["capture"]["workers_stopped"],
            }
            for c in cases
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cases"], indent=2))
    if not all(
        c["capture"]["workers_stopped"] and c["summary"]["measurement_complete"] for c in cases
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
