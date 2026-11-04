"""Synthetic active acquisition comparison; profile bounds are assumptions."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from embodied_agent.uncertainty_view import UncertaintyViewConfirmation

PROFILES = {
    "coarse": (0.004, 0.004, 0.004),
    "focus_x": (0.0004, 0.004, 0.004),
    "focus_y": (0.004, 0.0004, 0.004),
    "focus_z": (0.004, 0.004, 0.0004),
}
LIMITS = ((0.59, 0.69), (-0.04, 0.04), (0.032, 0.058))


def evaluate():
    rows = []
    for fault, axis, side, offset_mm, error_sign in itertools.product(
        ("nominal", "focus_missing", "false_precision"),
        range(3),
        (-1, 1),
        (-6, -1, 1),
        (-1, 0, 1),
    ):
        truth = [0.64, 0, 0.045]
        truth[axis] = LIMITS[axis][int(side == 1)] + side * offset_mm / 1000
        inside = all(low < p < high for p, (low, high) in zip(truth, LIMITS))
        results = {}
        for policy in ("coarse", "fixed_x", "active"):
            if policy == "active":
                check = UncertaintyViewConfirmation(0, PROFILES)
            else:
                initial = "coarse" if policy == "coarse" else "focus_x"
                check = UncertaintyViewConfirmation(
                    0, {initial: PROFILES[initial]}, initial=initial
                )

            def locate(
                profile, fault=fault, truth=truth, axis=axis, side=side, error_sign=error_sign
            ):
                if fault == "focus_missing" and profile != "coarse":
                    return None, None
                # New observations have profile-dependent errors. false_precision
                # intentionally breaks that contract, retaining coarse errors.
                actual_bounds = PROFILES["coarse" if fault == "false_precision" else profile]
                point = truth.copy()
                point[axis] += side * error_sign * actual_bounds[axis]
                return point, point

            for tick in range(16):
                check.step(tick * 0.2, locate)
            confirmed = check.status == "confirmed_in_tray"
            results[policy] = {
                "status": check.status,
                "confirmed": confirmed,
                "false_confirmation": confirmed and not inside,
                "switches": check.switches,
                "selected_profile": check.profile,
                "first_confirmation_time": next(
                    (
                        r["time"]
                        for r in check.rows
                        if r["observation"]["status"] == "confirmed_in_tray"
                    ),
                    None,
                ),
            }
        rows.append(
            {
                "fault": fault,
                "axis": axis,
                "side": side,
                "offset_mm": offset_mm,
                "error_sign": error_sign,
                "truth_inside_confirmation_region": inside,
                "policies": results,
            }
        )
    summary = {}
    for fault in ("nominal", "focus_missing", "false_precision"):
        subset = [r for r in rows if r["fault"] == fault]
        summary[fault] = {
            policy: {
                "cases": len(subset),
                "confirmed": sum(r["policies"][policy]["confirmed"] for r in subset),
                "false_confirmations": sum(
                    r["policies"][policy]["false_confirmation"] for r in subset
                ),
                "switched": sum(bool(r["policies"][policy]["switches"]) for r in subset),
            }
            for policy in ("coarse", "fixed_x", "active")
        }
    nominal = summary["nominal"]
    matches = (
        nominal["active"]["confirmed"]
        > nominal["fixed_x"]["confirmed"]
        > nominal["coarse"]["confirmed"]
        and nominal["active"]["false_confirmations"] == 0
        and summary["focus_missing"]["active"]["false_confirmations"] == 0
        and summary["false_precision"]["active"]["false_confirmations"] > 0
    )
    return {
        "schema_version": 1,
        "experiment": "synthetic_uncertainty_view_v1",
        "summary": summary,
        "matches_expected": matches,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [
        Path("scripts/evaluate_uncertainty_view.py"),
        *(
            Path("src/embodied_agent") / name
            for name in ("uncertainty_view.py", "cross_view.py", "release_confirmation.py")
        ),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = evaluate()
    if hashes != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise RuntimeError("source changed during evaluation")
    report["source_sha256"] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("cases", "source_sha256")}, indent=2
        )
    )
    return int(not report["matches_expected"])


if __name__ == "__main__":
    raise SystemExit(main())
