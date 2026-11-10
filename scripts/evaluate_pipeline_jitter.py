"""Synthetic exposure/processing jitter, keeping the confirmation rules frozen."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.pipelined_challenge import PipelinedChallengeController
from scripts.evaluate_pipelined_challenge import hashes

MODES = ("on_time", "request_5ms", "exposure_5ms", "candidate_5ms")
PROFILES = {"fixed80": (80, 80), "jitter80": (70, 90), "jitter100": (90, 110)}
SEEDS = (202611101, 202611102, 202611103, 202611104)


def sources():
    return {**hashes(), __file__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def run(mode, profile, capacity, seed, motion):
    c = PipelinedChallengeController(
        0,
        ExposureClockMap(0, 0),
        "synthetic",
        position_error_m=(0.001,) * 3,
        delivery_budget_s=0.1,
        exposure_lateness_s=0.01,
        capacity=capacity,
    )
    rng = random.Random(seed)
    delays = [rng.randint(*PROFILES[profile]) for _ in range(31)]
    jobs, trace = {}, []
    for ms in range(3101):
        now = ms / 1000
        for request, exposure_ms in jobs.pop(ms, []):
            position = (0.642 if motion and exposure_ms >= 2050 else 0.64, 0, 0.045)
            frame = StampedPosition(position, exposure_ms / 1000)
            c.complete(request.request_id, now, now, frame, frame, session_id="synthetic")
        if not (mode == "request_5ms" and 2050 <= ms < 2055):
            request = c.poll(now, now)
            if request:
                extra = (
                    5
                    if (
                        (mode == "exposure_5ms" and request.request_id == 12)
                        or (mode == "candidate_5ms" and request.request_id == 11)
                    )
                    else 0
                )
                exposure_ms = ms + extra
                delivery_ms = exposure_ms + delays[request.request_id - 1]
                jobs.setdefault(delivery_ms, []).append((request, exposure_ms))
                trace.append(
                    {
                        "id": request.request_id,
                        "request_ms": ms,
                        "exposure_ms": exposure_ms,
                        "delivery_ms": delivery_ms,
                    }
                )
        if c.status != "pending":
            break
    result = c.result()
    assert result["status"] != "pending"
    assert result["peak_inflight"] <= capacity
    assert result["request_count"] <= 31
    assert all(r["request_ms"] <= r["exposure_ms"] <= 3000 for r in trace)
    if motion or mode != "on_time":
        assert result["status"] != "confirmed_in_tray"
    if not motion and mode == "on_time" and profile in ("fixed80", "jitter80"):
        assert result["status"] == "confirmed_in_tray"
    return {
        "mode": mode,
        "profile": profile,
        "capacity": capacity,
        "seed": seed,
        "motion": motion,
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
        run(m, p, c, s, moving)
        for m in MODES
        for p in PROFILES
        for c in (2, 3)
        for s in SEEDS
        for moving in (False, True)
    ]
    assert sources() == before
    path = args.output / "report.json"
    path.write_text(
        json.dumps({"kind": "synthetic_virtual_time", "sources": before, "cases": cases}, indent=2)
        + "\n"
    )
    groups = []
    for mode in MODES:
        for profile in PROFILES:
            for capacity in (2, 3):
                subset = [
                    x
                    for x in cases
                    if (x["mode"], x["profile"], x["capacity"]) == (mode, profile, capacity)
                ]
                groups.append(
                    {
                        "mode": mode,
                        "profile": profile,
                        "capacity": capacity,
                        "static_reasons": dict(
                            Counter(x["result"]["reason"] for x in subset if not x["motion"])
                        ),
                        "motion_reasons": dict(
                            Counter(x["result"]["reason"] for x in subset if x["motion"])
                        ),
                    }
                )
    summary = {
        "kind": "synthetic_virtual_time",
        "case_count": len(cases),
        "seeds": SEEDS,
        "sources": before,
        "report_path": str(path),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "groups": groups,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(groups[:6], indent=2))
    print(f"Validated {len(cases)} cases: {path}")


if __name__ == "__main__":
    main()
