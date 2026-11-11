"""Synthetic two-stage confirmation; fast witness latency is an explicit assumption."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from embodied_agent.fresh_witness import FreshWitnessGate
from embodied_agent.observation_capture import StampedPosition
from scripts.evaluate_motion_shapes import ONSETS, PERIODS, SEEDS, WAVES, motion_value
from scripts.evaluate_sampling_aliases import run
from scripts.evaluate_sampling_aliases import sources as parent_sources


def evaluate(wave, onset, period, phase, seed, latency_ms):
    def motion(t):
        return 0 if wave == "static" else motion_value(t, wave, onset, period, phase)

    controllers = []
    baseline = run(
        "sweep",
        period,
        phase,
        seed,
        jitter_max_ms=2,
        motion=motion,
        on_controller=controllers.append,
    )
    result = {
        "wave": wave,
        "onset": onset,
        "period_ms": period,
        "phase": phase,
        "seed": seed,
        "witness_latency_ms": latency_ms,
        "baseline_outcome": baseline["outcome"],
        "baseline": baseline,
    }
    if controllers[0].status != "confirmed_in_tray":
        result["outcome"] = "candidate_rejected"
        return result
    gate = FreshWitnessGate(controllers[0])
    exposure = gate.start
    for _ in range(5):
        p = StampedPosition((0.64 + motion(exposure), 0, 0.045), exposure)
        delivery = exposure + latency_ms / 1000
        gate.deliver(delivery, delivery, p, p, session_id="audit")
        if gate.status != "pending":
            break
        exposure = delivery
    if gate.status == "pending":
        gate.tick(gate.deadline)
    snapshot = gate.result()
    result["gate"] = snapshot
    if gate.status == "confirmed_recent_evidence":
        end = gate.last_exposure
        truth = [motion(i / 1000) for i in range(2060, round(end * 1000) + 1)]
        span = max(truth) - min(truth)
        result["dense_truth_span_m"] = span
        result["outcome"] = (
            "false_stationarity_confirmation"
            if span > 0.001 + 1e-12
            else "recent_evidence_confirmation"
        )
        result["final_evidence_age_s"] = snapshot["decision_time_s"] - end
    else:
        result["outcome"] = gate.reason
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def sources():
        paths = (
            __file__,
            "src/embodied_agent/fresh_witness.py",
            "scripts/evaluate_motion_shapes.py",
        )
        return {
            **parent_sources(),
            **{p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        }

    before = sources()
    counts, witnesses, controls = {}, [], []
    n = 0
    path = args.output / "cases.jsonl"
    with path.open("w") as stream:
        for latency in (10, 80):
            for wave in WAVES:
                for onset in ONSETS:
                    counter = Counter()
                    counts[f"{latency}:{wave}:{onset}"] = counter
                    for period in PERIODS:
                        for phase in (0, 0.25, 0.5, 0.75):
                            for seed in SEEDS:
                                case = evaluate(wave, onset, period, phase, seed, latency)
                                stream.write(json.dumps(case) + "\n")
                                n += 1
                                counter[case["outcome"]] += 1
                                if case["outcome"] == "false_stationarity_confirmation":
                                    witnesses.append(
                                        {
                                            k: case[k]
                                            for k in (
                                                "wave",
                                                "onset",
                                                "period_ms",
                                                "phase",
                                                "seed",
                                                "witness_latency_ms",
                                                "outcome",
                                            )
                                        }
                                    )
            for seed in SEEDS:
                case = evaluate("static", 2.06, 55, 0, seed, latency)
                stream.write(json.dumps(case) + "\n")
                n += 1
                controls.append({"latency_ms": latency, "seed": seed, "outcome": case["outcome"]})
                assert case["outcome"] == (
                    "recent_evidence_confirmation" if latency == 10 else "stale_witness"
                )
    assert before == sources()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    summary = {
        "kind": "synthetic_fresh_witness",
        "case_count": n,
        "sources": before,
        "counts": counts,
        "controls": controls,
        "residual_misses": witnesses,
        "report_path": str(path),
        "report_sha256": digest.hexdigest(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"case_count": n, "counts": counts, "controls": controls}, indent=2))


if __name__ == "__main__":
    main()
