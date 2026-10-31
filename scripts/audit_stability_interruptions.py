"""Exhaustive discrete interruption sequences plus targeted recovery windows."""

import argparse
from collections import Counter
from itertools import product

from embodied_agent.exposure_confirmation import ExposureClockMap, ExposureConfirmation
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.recovery_report import write_recovery_report
from scripts.audit_release_stability import online_first_confirmation, reference_first_confirmation

POINTS = {"I": (0.64, 0, 0.033), "O": (0.64, 0, 0.032), "M": None}


def dual_first_confirmation(samples):
    # Binary-exact clock offset/delay removes conversion-roundoff from this audit.
    monitor = ExposureConfirmation(0, ExposureClockMap(16, 0))
    for exposure, point in samples:
        delivery = exposure + 1 / 64
        # Match the process observer's conservative delivery deadline.
        if delivery >= 3:
            return None
        observation = None if point is None else StampedPosition(point, 16 + exposure)
        monitor.step(delivery, 16 + delivery, observation)
        if monitor.status == "confirmed_in_tray":
            return exposure
        if monitor.status != "pending":
            return None
    return None


def targeted_cases():
    result = []
    for kind in ("outside", "missing", "jump"):
        for first in (1, 2, 3, 4, 7, 8):
            for length in (1, 2):
                samples = [(i / 4, POINTS["I"]) for i in range(13)]
                point = {"outside": POINTS["O"], "missing": None, "jump": (0.66, 0, 0.033)}[kind]
                for index in range(first, first + length):
                    samples[index] = (index / 4, point)
                result.append((f"{kind}_{first}_{length}", samples))
        for first in (7, 8):
            samples = [(i / 4, None if i < 5 else POINTS["I"]) for i in range(13)]
            point = {"outside": POINTS["O"], "missing": None, "jump": (0.66, 0, 0.033)}[kind]
            samples[first] = (first / 4, point)
            result.append((f"late_{kind}_{first}", samples))
    for gap in (0.25, 0.250001, 0.5):
        samples = [(0, POINTS["I"]), (0.25, POINTS["I"]), (0.5, POINTS["I"])]
        samples += [(0.5 + gap + i * 0.25, POINTS["I"]) for i in range(6)]
        result.append((f"gap_{gap}", samples))
    return result


def compare(samples):
    expected = reference_first_confirmation(samples)
    actual = online_first_confirmation(samples)
    delivered = [(time, point) for time, point in samples if time + 1 / 64 < 3]
    dual_expected = reference_first_confirmation(delivered)
    dual_actual = dual_first_confirmation(samples)
    return {
        "expected_first_confirmation": expected,
        "actual_first_confirmation": actual,
        "dual_expected": dual_expected,
        "dual_actual": dual_actual,
        "match": expected == actual and dual_expected == dual_actual,
    }


def evaluate():
    failures, counts = [], Counter()
    for word in product("IOM", repeat=9):
        samples = [(i / 4, POINTS[symbol]) for i, symbol in enumerate(word)]
        row = compare(samples)
        counts[
            "confirmed" if row["actual_first_confirmation"] is not None else "not_confirmed"
        ] += 1
        if not row["match"]:
            failures.append({"sequence": "".join(word), **row})
    targeted = [
        {"case": name, "samples": samples, **compare(samples)} for name, samples in targeted_cases()
    ]
    return {
        "schema_version": 1,
        "algorithm": "exposure_confirmation_v2",
        "alphabet": {"I": "inside", "O": "excluded_boundary", "M": "missing"},
        "interval": 0.25,
        "sequence_length": 9,
        "exhaustive_cases": 3**9,
        "exhaustive_outcomes": dict(counts),
        "exhaustive_failures": failures,
        "targeted_cases": len(targeted),
        "targeted_results": targeted,
        "mismatches": len(failures) + sum(not row["match"] for row in targeted),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate()
    write_recovery_report(args.output, report)
    print(
        f"{report['exhaustive_cases']} exhaustive + {report['targeted_cases']} targeted; "
        f"{report['mismatches']} mismatches"
    )
    return int(report["mismatches"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
