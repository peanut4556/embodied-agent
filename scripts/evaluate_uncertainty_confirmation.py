"""Deterministic boundary stress grid; truth is used only for scoring."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from embodied_agent.rigid_reference import RigidReferenceConfirmation

REFERENCES = {
    "blue": (0.48, -0.12, 0.004),
    "yellow": (0.64, -0.12, 0.004),
    "magenta": (0.80, 0.12, 0.004),
}
CHECKPOINT = (0.48, 0.06, 0.034)
# Independent scoring specification: the conservative confirmation region,
# not PhysicsWorld.inside_box's wider task goal.
LIMITS = ((0.59, 0.69), (-0.04, 0.04), (0.032, 0.058))


def evaluate():
    rows = []
    angle = np.deg2rad(2)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    center = np.mean(list(REFERENCES.values()), axis=0)

    def measured(point):
        return rotation @ (np.array(point) - center) + center + (-0.01, 0, 0)

    refs = {
        name: measured(point) for name, point in {**REFERENCES, "checkpoint": CHECKPOINT}.items()
    }
    for axis, side, offset_mm, error_mm, budget_mm in itertools.product(
        range(3), (-1, 1), (-6, -1, 1), (-4, 0, 4), (1, 4)
    ):
        truth = np.array([0.64, 0, 0.045])
        truth[axis] = LIMITS[axis][int(side == 1)] + side * offset_mm / 1000
        estimate = truth.copy()
        estimate[axis] += side * error_mm / 1000
        point = measured(estimate)
        baseline = RigidReferenceConfirmation(0, REFERENCES, CHECKPOINT)
        guarded = RigidReferenceConfirmation(
            0, REFERENCES, CHECKPOINT, position_error_m=(budget_mm / 1000,) * 3
        )
        # Identical shared target error survives a perfect reference fit/check.
        for tick in range(16):
            for check in (baseline, guarded):
                check.step(tick * 0.2, point, point, refs, refs)
        inside = all(low < value < high for value, (low, high) in zip(truth, LIMITS))
        old = baseline.status == "confirmed_in_tray"
        new = guarded.status == "confirmed_in_tray"
        rows.append(
            {
                "axis": axis,
                "side": side,
                "offset_mm": offset_mm,
                "error_mm": error_mm,
                "budget_mm": budget_mm,
                "budget_covers_error": abs(error_mm) <= budget_mm,
                "truth_inside_confirmation_region": inside,
                "baseline_confirmed": old,
                "guarded_confirmed": new,
                "baseline_false": old and not inside,
                "guarded_false": new and not inside,
                "guarded_status": guarded.status,
                "first_gate": guarded.result()["paired"]["observations"][0]["gate"],
            }
        )
    summary = {}
    for budget in (1, 4):
        subset = [r for r in rows if r["budget_mm"] == budget]
        summary[str(budget)] = {
            "cases": len(subset),
            "covered_cases": sum(r["budget_covers_error"] for r in subset),
            "baseline_confirmed": sum(r["baseline_confirmed"] for r in subset),
            "guarded_confirmed": sum(r["guarded_confirmed"] for r in subset),
            "baseline_false": sum(r["baseline_false"] for r in subset),
            "guarded_false": sum(r["guarded_false"] for r in subset),
            "lost_true_confirmations": sum(
                r["truth_inside_confirmation_region"]
                and r["baseline_confirmed"]
                and not r["guarded_confirmed"]
                for r in subset
            ),
        }
    matches = (
        not any(r["guarded_false"] and r["budget_covers_error"] for r in rows)
        and summary["1"]["guarded_false"] > 0
        and summary["4"]["guarded_confirmed"] > 0
        and summary["4"]["lost_true_confirmations"] > 0
    )
    return {
        "schema_version": 1,
        "experiment": "synthetic_rigid_boundary_grid_v1",
        "summary_by_budget_mm": summary,
        "matches_expected": matches,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [
        Path(__file__),
        *(
            Path("src/embodied_agent") / name
            for name in (
                "rigid_reference.py",
                "reference_confirmation.py",
                "cross_view.py",
                "release_confirmation.py",
            )
        ),
    ]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report = evaluate()
    if hashes != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}:
        raise RuntimeError("source changed during evaluation")
    report["source_sha256"] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("cases", "source_sha256")},
            indent=2,
        )
    )
    return int(not report["matches_expected"])


if __name__ == "__main__":
    raise SystemExit(main())
