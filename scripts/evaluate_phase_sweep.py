"""Frozen phase sweep evaluated on known failures and a separate offset grid."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.evaluate_sampling_aliases import run
from scripts.evaluate_sampling_aliases import sources as parent_sources
from scripts.evaluate_unequal_sampling import PERIODS_MS
from scripts.evaluate_unequal_sampling import SEEDS as DESIGN_SEEDS

# No periods or phases in common with the previous design grid.
HOLDOUT_PERIODS_MS = tuple(x + 2.5 for x in range(20, 240, 5) if x != 25)
HOLDOUT_PHASES = tuple((i + 0.5) / 16 for i in range(16))
SEEDS = (None, 202611119, 202611120)
FAILURES = "docs/evaluations/unequal-sampling-v1.json"


def sources():
    paths = (__file__, FAILURES, "scripts/evaluate_unequal_sampling.py")
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
    known = [
        x for x in json.loads(Path(FAILURES).read_text())["misses"] if x["profile"] == "unequal"
    ]
    counters, misses, controls = {}, [], []
    path = args.output / "cases.jsonl"
    count = 0
    with path.open("w") as stream:

        def record(profile, period, phase, seed, split, bound=2):
            nonlocal count
            case = run(profile, period, phase, seed, jitter_max_ms=bound)
            case["split"] = split
            stream.write(json.dumps(case) + "\n")
            count += 1
            key = f"{split}:{profile}:{seed}"
            counters.setdefault(key, Counter())[case["outcome"]] += 1
            if case["outcome"] == "false_stationarity_confirmation":
                misses.append(
                    {
                        k: case[k]
                        for k in (
                            "split",
                            "profile",
                            "period_ms",
                            "phase_cycles",
                            "jitter_seed",
                            "dense_truth_span_m",
                            "observed_challenge_span_m",
                        )
                    }
                )
            if period is None:
                controls.append(
                    {
                        "profile": profile,
                        "seed": seed,
                        "bound_ms": bound,
                        "outcome": case["outcome"],
                        "reason": case["reason"],
                    }
                )
                if bound == 2:
                    assert case["outcome"] == "static_confirmation"

        for profile in ("unequal", "sweep"):
            for case in known:
                record(
                    profile, case["period_ms"], case["phase_cycles"], case["jitter_seed"], "known"
                )
            for seed in DESIGN_SEEDS:
                for period in PERIODS_MS:
                    for phase in range(16):
                        record(profile, period, phase / 16, seed, "reference")
            for seed in SEEDS:
                for period in HOLDOUT_PERIODS_MS:
                    for phase in HOLDOUT_PHASES:
                        record(profile, period, phase, seed, "holdout")
                for bound in (2, 5):
                    record(profile, None, 0, seed, "control", bound)
    assert before == sources()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    summary = {
        "kind": "synthetic_phase_sweep_holdout",
        "case_count": count,
        "holdout_periods_ms": HOLDOUT_PERIODS_MS,
        "holdout_phase_cycles": HOLDOUT_PHASES,
        "seeds": SEEDS,
        "reference_periods_ms": PERIODS_MS,
        "reference_seeds": DESIGN_SEEDS,
        "sources": before,
        "counts": counters,
        "misses": misses,
        "controls": controls,
        "report_path": str(path),
        "report_sha256": digest.hexdigest(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"case_count": count, "counts": counters}, indent=2))


if __name__ == "__main__":
    main()
