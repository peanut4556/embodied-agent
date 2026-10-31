"""Compare online stability against exhaustive valid-window search (no state reuse)."""

import argparse
import math
import random
from itertools import pairwise

from embodied_agent.recovery_report import write_recovery_report
from embodied_agent.release_confirmation import ReleaseConfirmation


def reference_first_confirmation(samples):
    # Independent offline formulation: enumerate complete in-region windows.
    # No ReleaseConfirmation state or internal helper is used here.
    for end, (time, _) in enumerate(samples):
        if time > 3 + 1e-9:
            return None
        for begin in range(end + 1):
            window = samples[begin : end + 1]
            if time - window[0][0] < 1 - 1e-9:
                continue
            if any(
                point is None
                or not (
                    abs(point[0] - 0.64) < 0.05
                    and abs(point[1]) < 0.04
                    and 0.032 < point[2] < 0.058
                )
                for _, point in window
            ):
                continue
            if all(
                0 < t2 - t1 <= 0.25 and math.dist(p1, p2) / (t2 - t1) < 0.02
                for (t1, p1), (t2, p2) in pairwise(window)
            ):
                return time
        if time >= 3 - 1e-9:
            return None
    return None


def online_first_confirmation(samples):
    monitor = ReleaseConfirmation(0)
    for time, point in samples:
        monitor.observe(time, point)
        if monitor.status == "confirmed_in_tray":
            return time
        if monitor.status != "pending":
            return None
    return None


def cases(seed=20261031, count=400):
    result = []
    # Binary-exact times isolate the outside-to-inside timing defect.
    for axis, outside, inside in (
        (2, 0.032, 0.033),
        (2, 0.058, 0.057),
        (1, -0.04, -0.039),
        (1, 0.04, 0.039),
    ):
        samples = []
        for i in range(6):
            point = [0.64, 0, 0.04]
            point[axis] = outside if i == 0 else inside
            samples.append((i * 0.25, point))
        result.append((f"boundary_entry_{axis}_{outside}", samples))
    for gap in (0.249, 0.25, 0.251):
        result.append((f"gap_{gap}", [(i * gap, [0.64, 0, 0.04]) for i in range(9)]))
    for speed in (0.0199, 0.02, 0.0201):
        result.append(
            (f"speed_{speed}", [(i * 0.25, [0.62 + speed * i * 0.25, 0, 0.04]) for i in range(9)])
        )
    for end in (0.999, 1, 1.001):
        result.append((f"duration_{end}", [(i * end / 5, [0.64, 0, 0.04]) for i in range(6)]))
    rng = random.Random(seed)
    for index in range(count):
        samples, time = [], 0
        point = [0.64, 0, 0.04]
        for _ in range(14):
            if time > 3.1:
                break
            mode = rng.randrange(8)
            if mode == 0:
                value = None
            elif mode == 1:
                value = [0.64, 0, rng.choice((0.032, 0.058))]
            else:
                if mode == 2:
                    point = [0.64, 0, rng.choice((0.033, 0.04, 0.057))]
                elif mode == 3:
                    point = [point[0] + rng.uniform(-0.008, 0.008), point[1], point[2]]
                value = list(point)
            samples.append((time, value))
            time = round(time + rng.choice((0.125, 0.2, 0.25, 0.251)), 6)
        result.append((f"random_{index}", samples))
    return result


def evaluate():
    rows = []
    for name, samples in cases():
        expected = reference_first_confirmation(samples)
        actual = online_first_confirmation(samples)
        rows.append(
            {
                "case": name,
                "expected_first_confirmation": expected,
                "actual_first_confirmation": actual,
                "match": actual == expected,
                **(
                    {"samples": samples}
                    if not name.startswith("random_") or actual != expected
                    else {}
                ),
            }
        )
    return {
        "schema_version": 1,
        "seed": 20261031,
        "algorithm": "exposure_confirmation_v2",
        "cases": len(rows),
        "mismatches": sum(not row["match"] for row in rows),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate()
    write_recovery_report(args.output, report)
    print(f"{report['cases']} cases; {report['mismatches']} mismatches")
    return int(report["mismatches"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
