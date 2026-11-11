"""Expanded paired alias grid with matched 0--2 ms lateness and static controls."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.evaluate_sampling_aliases import run
from scripts.evaluate_sampling_aliases import sources as parent_sources

PERIODS_MS = tuple(sorted({*range(20, 241, 5), 27.5}))
SEEDS = (None, 202611117, 202611118)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def sources():
        return {
            **parent_sources(),
            __file__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    before = sources()
    counts, misses, controls = {}, [], []
    path = args.output / "cases.jsonl"
    n = 0
    with path.open("w") as stream:
        for profile in ("spaced", "unequal"):
            for seed in SEEDS:
                counter = Counter()
                counts[f"{profile}:{seed}"] = counter
                for period in PERIODS_MS:
                    for phase in range(16):
                        case = run(profile, period, phase / 16, seed, jitter_max_ms=2)
                        stream.write(json.dumps(case) + "\n")
                        n += 1
                        counter[case["outcome"]] += 1
                        if case["outcome"] == "false_stationarity_confirmation":
                            misses.append(
                                {
                                    k: case[k]
                                    for k in (
                                        "profile",
                                        "period_ms",
                                        "phase_cycles",
                                        "jitter_seed",
                                        "dense_truth_span_m",
                                        "observed_challenge_span_m",
                                    )
                                }
                            )
                for bound in (2, 5):
                    case = run(profile, None, 0, seed, jitter_max_ms=bound)
                    stream.write(json.dumps(case) + "\n")
                    n += 1
                    controls.append(
                        {
                            "profile": profile,
                            "seed": seed,
                            "lateness_bound_ms": bound,
                            "outcome": case["outcome"],
                            "reason": case["reason"],
                        }
                    )
                    if bound == 2:
                        assert case["outcome"] == "static_confirmation"
    assert before == sources()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    summary = {
        "kind": "synthetic_paired_unequal_sampling",
        "case_count": n,
        "periods_ms": PERIODS_MS,
        "phase_cycles": [i / 16 for i in range(16)],
        "jitter_seeds": SEEDS,
        "motion_lateness_bound_ms": 2,
        "sources": before,
        "counts": counts,
        "misses": misses,
        "controls": controls,
        "report_path": str(path),
        "report_sha256": digest.hexdigest(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"case_count": n, "counts": counts, "controls": controls}, indent=2))


if __name__ == "__main__":
    main()
