"""Synthetic frequency/phase audit; truth is used only for offline scoring."""

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from embodied_agent.aperiodic_challenge import AperiodicChallengeController
from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.phase_sweep_challenge import PhaseSweepChallengeController
from embodied_agent.pipelined_challenge import PipelinedChallengeController
from embodied_agent.spaced_challenge import SpacedChallengeController
from scripts.evaluate_spaced_challenge import sources as parent_sources

PERIODS_MS = (25, 27.5, 50, 55, 100, 110, 200, 220)
JITTER_SEEDS = (None, 202611115, 202611116)


def displacement(t, period_ms, phase):
    if period_ms is None or t < 2.06:
        return 0.0
    return 0.002 * math.sin(2 * math.pi * ((t - 2.05) * 1000 / period_ms + phase))


def run(
    profile, period_ms, phase, jitter_seed, *, jitter_max_ms=5, motion=None, on_controller=None
):
    sample = motion if motion is not None else lambda t: displacement(t, period_ms, phase)
    kwargs = {"position_error_m": (0.001,) * 3, "capacity": 2, "delivery_budget_s": 0.1}
    if profile == "strict":
        c = PipelinedChallengeController(
            0, ExposureClockMap(0, 0), "audit", exposure_lateness_s=0.005, **kwargs
        )
    elif profile == "spaced":
        c = SpacedChallengeController(0, ExposureClockMap(0, 0), "audit", **kwargs)
    elif profile == "unequal":
        c = AperiodicChallengeController(0, ExposureClockMap(0, 0), "audit", **kwargs)
    elif profile == "sweep":
        c = PhaseSweepChallengeController(0, ExposureClockMap(0, 0), "audit", **kwargs)
    else:
        raise ValueError("unknown profile")
    rng = random.Random(jitter_seed)
    jitter = [0 if jitter_seed is None else rng.randint(0, jitter_max_ms) for _ in range(31)]
    jobs, frames = {}, []
    for ms in range(3261):
        now = ms / 1000
        for req, exposure in jobs.pop(ms, []):
            p = StampedPosition((0.64 + sample(exposure / 1000), 0, 0.045), exposure / 1000)
            c.complete(req.request_id, now, now, p, p, session_id="audit")
        req = c.poll(now, now)
        if req:
            exposure = ms + jitter[req.request_id - 1]
            jobs.setdefault(exposure + 80, []).append((req, exposure))
            frames.append(
                {
                    "request_id": req.request_id,
                    "request_ms": ms,
                    "exposure_ms": exposure,
                    "delivery_ms": exposure + 80,
                }
            )
        if c.status != "pending":
            break
    result = c.result()
    assert result["status"] != "pending"
    assert result["peak_inflight"] <= 2 and result["request_count"] <= 31
    # Score actual observation time, not delayed callback time. This independent
    # dense trace is never supplied to the controller. It is not physics truth.
    exposures = [d["exposure_time"] for d in c.check.rows if d["exposure_time"] is not None]
    end = max(exposures, default=0)
    truth = [sample(i / 1000) for i in range(2060, round(end * 1000) + 1)]
    truth_span = max(truth) - min(truth) if truth else 0
    confirmed = c.status == "confirmed_in_tray"
    if confirmed:
        outcome = (
            "false_stationarity_confirmation"
            if truth_span > 0.001 + 1e-12
            else "static_confirmation"
        )
    elif c.reason == "challenge_motion":
        outcome = "motion_rejection"
    else:
        outcome = "timing_or_other_rejection"
    evidence = c.check.evidence.result()
    spans = [
        x["raw_axis_span_m"] for x in evidence["observations"] if x["raw_axis_span_m"] is not None
    ]
    if on_controller is not None:
        on_controller(c)
    return {
        "profile": profile,
        "period_ms": period_ms,
        "phase_cycles": phase,
        "jitter_seed": jitter_seed,
        "jitter_max_ms": jitter_max_ms,
        "outcome": outcome,
        "reason": c.reason,
        "dense_truth_span_m": truth_span,
        "observed_challenge_span_m": max(spans, default=None),
        "last_exposure_s": end,
        "frames": frames,
        "result": result,
    }


def sources():
    paths = (
        __file__,
        "src/embodied_agent/aperiodic_challenge.py",
        "src/embodied_agent/phase_sweep_challenge.py",
    )
    return {
        **parent_sources(),
        **{p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = sources()
    cases = [
        run(p, period, phase / 8, seed)
        for p in ("strict", "spaced")
        for period in PERIODS_MS
        for phase in range(8)
        for seed in JITTER_SEEDS
    ]
    cases += [run(p, None, 0, seed) for p in ("strict", "spaced") for seed in JITTER_SEEDS]
    assert sources() == before
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps({"kind": "synthetic_alias_audit", "sources": before, "cases": cases}, indent=2)
        + "\n"
    )
    groups = []
    for p in ("strict", "spaced"):
        for seed in JITTER_SEEDS:
            subset = [
                x
                for x in cases
                if x["profile"] == p and x["jitter_seed"] == seed and x["period_ms"] is not None
            ]
            groups.append(
                {
                    "profile": p,
                    "jitter_seed": seed,
                    "outcomes": dict(Counter(x["outcome"] for x in subset)),
                }
            )
    keys = (
        "profile",
        "period_ms",
        "phase_cycles",
        "jitter_seed",
        "dense_truth_span_m",
        "observed_challenge_span_m",
        "last_exposure_s",
    )
    summary = {
        "kind": "synthetic_alias_audit",
        "case_count": len(cases),
        "sources": before,
        "periods_ms": PERIODS_MS,
        "phase_cycles": [i / 8 for i in range(8)],
        "groups": groups,
        "misses": [
            {k: x[k] for k in keys}
            for x in cases
            if x["outcome"] == "false_stationarity_confirmation"
        ],
        "controls": [
            {"profile": x["profile"], "jitter_seed": x["jitter_seed"], "outcome": x["outcome"]}
            for x in cases
            if x["period_ms"] is None
        ],
        "report_path": str(report_path),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(groups, indent=2))
    print(f"Saved {len(cases)} cases to {report_path}")


if __name__ == "__main__":
    main()
