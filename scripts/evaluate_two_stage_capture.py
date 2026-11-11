"""Causal two-stage driver simulation with failure injection, no real camera."""

import argparse
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.two_stage_capture import TwoStageCaptureController
from scripts.evaluate_fresh_witness import parent_sources

SERVICES = (1, 10, 20, 21, 80, "stall", "cached")


def run(scene, service, seed):
    c = TwoStageCaptureController(
        0, ExposureClockMap(0, 0), "camera", position_error_m=(0.001,) * 3
    )
    rng = random.Random(seed)
    delays = [rng.randint(0, 2) for _ in range(31)]
    jobs, trace = {}, []
    peak_witness = 0
    for ms in range(3401):
        now = ms / 1000
        for req, exposure in jobs.pop(ms, []):
            t = exposure / 1000
            shift = 0
            if t >= 3.17:
                if scene == "step":
                    shift = 0.002
                elif scene == "sine":
                    shift = 0.002 * math.sin(2 * math.pi * (t - 3.17) / 0.055)
                elif scene == "pulse":
                    shift = 0.002 if ((t - 3.17) / 0.055) % 1 < 0.1 else 0
            stamp = (req.host_time - 0.001) if service == "cached" and req.stage == "witness" else t
            p = StampedPosition((0.64 + shift, 0, 0.045), stamp)
            c.complete(req.request_id, now, now, p, p, session_id="camera")
            if req.request_id == 31:
                assert c.status == "pending" and c.stage == "witness"
        req = c.poll(now, now)
        if req:
            exposure = ms + (delays[req.request_id - 1] if req.stage == "candidate" else 0)
            latency = 80 if req.stage == "candidate" else (10 if service == "cached" else service)
            delivery = None if latency == "stall" else exposure + latency
            if delivery is not None:
                jobs.setdefault(delivery, []).append((req, exposure))
            trace.append(
                {
                    "request": req.request_id,
                    "stage": req.stage,
                    "request_ms": ms,
                    "exposure_ms": exposure,
                    "planned_delivery_ms": delivery,
                }
            )
        peak_witness = max(peak_witness, sum(r.stage == "witness" for r, _ in c.pending.values()))
        if c.status != "pending":
            break
    result = c.result()
    assert c.status != "pending" and len(c.requests) <= 36 and peak_witness <= 1
    witnesses = [r for r in trace if r["stage"] == "witness"]
    for previous, current in itertools.pairwise(witnesses):
        assert previous["planned_delivery_ms"] <= current["request_ms"]
    if scene == "static":
        assert (c.status == "confirmed_recent_evidence") == (service in (1, 10, 20))
    return {
        "scene": scene,
        "service_ms": service,
        "seed": seed,
        "peak_witness_inflight": peak_witness,
        "trace": trace,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def sources():
        paths = (
            __file__,
            "src/embodied_agent/two_stage_capture.py",
            "src/embodied_agent/fresh_witness.py",
            "scripts/evaluate_fresh_witness.py",
        )
        return {
            **parent_sources(),
            **{p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        }

    before = sources()
    cases = [
        run(scene, service, seed)
        for scene in ("static", "step", "sine", "pulse")
        for service in SERVICES
        for seed in (202611123, 202611124)
    ]
    assert sources() == before
    path = args.output / "report.json"
    path.write_text(
        json.dumps(
            {"kind": "synthetic_causal_two_stage", "sources": before, "cases": cases}, indent=2
        )
        + "\n"
    )
    summary = {
        "kind": "synthetic_causal_two_stage",
        "sources": before,
        "case_count": len(cases),
        "report_path": str(path),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cases": [
            {
                "scene": c["scene"],
                "service_ms": c["service_ms"],
                "seed": c["seed"],
                "status": c["result"]["status"],
                "reason": c["result"]["reason"],
                "requests": len(c["result"]["requests"]),
                "cancel_required": c["result"]["driver_cancel_required"],
            }
            for c in cases
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Validated {len(cases)} causal runs")


if __name__ == "__main__":
    main()
