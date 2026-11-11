"""Paired synthetic jitter experiments; not a camera throughput measurement."""

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.pipelined_challenge import PipelinedChallengeController
from embodied_agent.spaced_challenge import SpacedChallengeController
from scripts.evaluate_pipelined_challenge import hashes


def sources():
    paths = (__file__, "src/embodied_agent/spaced_challenge.py")
    return {**hashes(), **{p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}


def run(profile, jitter, scene, seed):
    rng = random.Random(seed)
    lateness = {
        "zero": [0] * 31,
        "alternating": [5 * (i % 2) for i in range(31)],
        "random": [rng.randint(0, 5) for _ in range(31)],
        "constant5": [5] * 31,
    }[jitter]
    delays = [rng.randint(70, 90) for _ in range(31)]
    kwargs = {"position_error_m": (0.001,) * 3, "capacity": 2, "delivery_budget_s": 0.1}
    if profile == "strict":
        c = PipelinedChallengeController(
            0, ExposureClockMap(0, 0), "sim", exposure_lateness_s=0.005, **kwargs
        )
    else:
        c = SpacedChallengeController(0, ExposureClockMap(0, 0), "sim", **kwargs)
    jobs, trace = {}, []
    # The controller is polled at each planned request and every millisecond.
    # Exposure occurs later, before its callback; the actual timestamp is retained.
    for ms in range(3261):
        now = ms / 1000
        for req, exposure in jobs.pop(ms, []):
            t = exposure / 1000
            shift = 0
            if scene == "step" and t >= 2.1:
                shift = 0.002
            elif scene == "periodic" or (scene == "periodic_after" and t >= 2.1):
                shift = 0.002 * math.sin(2 * math.pi * t / 0.2)
            p = StampedPosition((0.64 + shift, 0, 0.045), t)
            c.complete(req.request_id, now, now, p, p, session_id="sim")
        req = c.poll(now, now)
        if req:
            exposure = ms + lateness[req.request_id - 1]
            delivery = exposure + delays[req.request_id - 1]
            jobs.setdefault(delivery, []).append((req, exposure))
            trace.append(
                {
                    "id": req.request_id,
                    "request_ms": ms,
                    "exposure_ms": exposure,
                    "delivery_ms": delivery,
                }
            )
        if c.status != "pending":
            break
    result = c.result()
    assert c.status != "pending"
    assert result["peak_inflight"] <= 2 and result["request_count"] <= 31
    if c.status == "confirmed_in_tray":
        evidence = result["confirmation"]["evidence"]
        assert len(evidence["observations"]) == 31
        assert evidence["observations"][-1]["time"] - evidence["candidate_time"] >= 1 - 1e-9
        assert all(x["exposure_ms"] <= round(c.check.capture_deadline * 1000) for x in trace)
    if profile == "spaced" and scene == "static":
        assert c.status == "confirmed_in_tray", result["reason"]
    if scene in ("step", "periodic_after"):
        assert c.status != "confirmed_in_tray"
    return {
        "profile": profile,
        "jitter": jitter,
        "scene": scene,
        "seed": seed,
        "trace": trace,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = sources()
    cases = [
        run(p, j, s, seed)
        for p in ("strict", "spaced")
        for j in ("zero", "alternating", "random", "constant5")
        for s in ("static", "step", "periodic", "periodic_after")
        for seed in (202611111, 202611112, 202611113, 202611114)
    ]
    assert sources() == before
    path = args.output / "report.json"
    path.write_text(
        json.dumps({"kind": "synthetic_virtual_time", "sources": before, "cases": cases}, indent=2)
        + "\n"
    )
    groups = []
    for p in ("strict", "spaced"):
        for j in ("zero", "alternating", "random", "constant5"):
            for s in ("static", "step", "periodic", "periodic_after"):
                rows = [x for x in cases if (x["profile"], x["jitter"], x["scene"]) == (p, j, s)]
                groups.append(
                    {
                        "profile": p,
                        "jitter": j,
                        "scene": s,
                        "reasons": dict(Counter(x["result"]["reason"] for x in rows)),
                    }
                )
    summary = {
        "kind": "synthetic_virtual_time",
        "case_count": len(cases),
        "sources": before,
        "report_path": str(path),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "groups": groups,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(groups, indent=2))
    print(f"Validated {len(cases)} cases: {path}")


if __name__ == "__main__":
    main()
