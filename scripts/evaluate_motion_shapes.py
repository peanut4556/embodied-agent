"""Waveform/onset stress and a separate final-exposure-to-delivery blind-window audit."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts.evaluate_sampling_aliases import run
from scripts.evaluate_sampling_aliases import sources as parent_sources

WAVES = ("sine", "triangle", "pulse")
ONSETS = (2.06, 2.5, 3.10, 3.17)
PERIODS = (55, 83, 110, 197)
SEEDS = (None, 202611121, 202611122)


def motion_value(t, wave, onset, period_ms, phase):
    if t < onset:
        return 0.0
    cycle = ((t - onset) * 1000 / period_ms + phase) % 1
    if wave == "sine":
        return 0.002 * math.sin(2 * math.pi * cycle)
    if wave == "triangle":
        return 0.002 * (1 - 4 * abs(cycle - 0.5))
    if wave == "pulse":
        return 0.002 if cycle < 0.1 - 1e-10 else 0.0
    raise ValueError("unknown waveform")


def evaluate(profile, wave, onset, period, phase, seed):
    def motion(t):
        return motion_value(t, wave, onset, period, phase)

    case = run(profile, period, phase, seed, jitter_max_ms=2, motion=motion)
    case.update(wave=wave, onset_s=onset)
    last = case["last_exposure_s"]
    deliveries = case["result"]["confirmation"]["deliveries"]
    end = deliveries[-1]["delivery_time"] if deliveries else last
    blind = [motion(i / 1000) for i in range(round(last * 1000), round(end * 1000) + 1)]
    span = max(blind) - min(blind) if blind else 0
    case["post_exposure_span_m"] = span
    case["confirmed_with_post_exposure_motion"] = (
        case["result"]["status"] == "confirmed_in_tray" and span > 0.001 + 1e-12
    )
    return case


def sources():
    return {**parent_sources(), __file__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = sources()
    counts, witnesses, controls = {}, [], []
    total = 0
    path = args.output / "cases.jsonl"
    with path.open("w") as stream:
        for profile in ("unequal", "sweep"):
            for wave in WAVES:
                for onset in ONSETS:
                    counter = Counter()
                    counts[f"{profile}:{wave}:{onset}"] = counter
                    for period in PERIODS:
                        for phase in (0, 0.25, 0.5, 0.75):
                            for seed in SEEDS:
                                case = evaluate(profile, wave, onset, period, phase, seed)
                                stream.write(json.dumps(case) + "\n")
                                total += 1
                                counter[case["outcome"]] += 1
                                if case["confirmed_with_post_exposure_motion"]:
                                    counter["post_exposure_motion_overlap"] += 1
                                if (
                                    case["outcome"] == "false_stationarity_confirmation"
                                    or case["confirmed_with_post_exposure_motion"]
                                ):
                                    witnesses.append(
                                        {
                                            k: case[k]
                                            for k in (
                                                "profile",
                                                "wave",
                                                "onset_s",
                                                "period_ms",
                                                "phase_cycles",
                                                "jitter_seed",
                                                "outcome",
                                                "dense_truth_span_m",
                                                "last_exposure_s",
                                                "post_exposure_span_m",
                                                "confirmed_with_post_exposure_motion",
                                            )
                                        }
                                    )
            for seed in SEEDS:
                case = run(profile, None, 0, seed, jitter_max_ms=2)
                assert case["outcome"] == "static_confirmation"
                stream.write(json.dumps(case) + "\n")
                total += 1
                controls.append({"profile": profile, "seed": seed, "outcome": case["outcome"]})
    assert sources() == before
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    summary = {
        "kind": "synthetic_waveform_and_delivery_gap_audit",
        "case_count": total,
        "sources": before,
        "seeds": SEEDS,
        "periods_ms": PERIODS,
        "onsets_s": ONSETS,
        "phases": [0, 0.25, 0.5, 0.75],
        "counts": counts,
        "controls": controls,
        "witnesses": witnesses,
        "report_path": str(path),
        "report_sha256": digest.hexdigest(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"case_count": total, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
