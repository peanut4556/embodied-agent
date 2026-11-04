"""Freeze rendered calibration budgets, then compare fresh held-out acquisitions."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.empirical_bounds import estimate_profile_bounds
from embodied_agent.rigid_reference import RigidReferenceConfirmation
from embodied_agent.uncertainty_view import UncertaintyViewConfirmation
from scripts.evaluate_rigid_reference import CHECKPOINT, REFERENCES, capture, make_world

RESOLUTIONS = {"coarse": (240, 320), "detail": (480, 640)}
CALIBRATION = [(x, y) for x in (0.615, 0.64, 0.665) for y in (-0.02, 0.02)]
VALIDATION = [(x, 0.001) for x in (0.589, 0.5915, 0.5925, 0.595, 0.685, 0.6875, 0.6885, 0.691)] + [
    (0.641, y) for y in (-0.041, -0.0385, 0.0385, 0.041)
]


def inside(point):
    return 0.59 < point[0] < 0.69 and -0.04 < point[1] < 0.04 and 0.032 < point[2] < 0.058


def place(world, xy):
    world.reset()
    world.data.qpos[5:8] = [*xy, 0.12]
    world.data.qpos[8:12] = [1, 0, 0, 0]
    world.data.qvel[:] = 0
    mujoco.mj_forward(world.model, world.data)
    for _ in range(500):
        mujoco.mj_step(world.model, world.data)


def acquire(world, renderer, correction):
    world.camera_renderer = renderer
    packet = {}
    for key, side in (("primary", 1), ("secondary", -1)):
        raw, refs = capture(world, side, "nominal")
        corrected, diagnostic = correction.correct(raw, refs)
        packet[key] = corrected
        packet[key + "_correction"] = diagnostic
    return packet


def sources():
    return [
        Path(p)
        for p in (
            "scripts/evaluate_rendered_precision.py",
            "scripts/evaluate_rigid_reference.py",
            "scripts/evaluate_reference_confirmation.py",
            *(
                f"src/embodied_agent/{name}"
                for name in (
                    "empirical_bounds.py",
                    "uncertainty_view.py",
                    "rigid_reference.py",
                    "reference_confirmation.py",
                    "reference_perception.py",
                    "cross_view.py",
                    "release_confirmation.py",
                    "physics.py",
                    "perception.py",
                    "assets/tabletop.xml",
                )
            ),
        )
    ]


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources()}
    report = {
        "schema_version": 1,
        "status": "running",
        "source_sha256": hashes,
        "resolutions_hw": RESOLUTIONS,
        "calibration_positions": CALIBRATION,
        "validation_positions": VALIDATION,
        "padding_m": 0.0005,
        "calibration": [],
        "cases": [],
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    world, renderers = None, {}
    try:
        world = make_world("nominal")
        renderers = {
            key: mujoco.Renderer(world.model, height=h, width=w)
            for key, (h, w) in RESOLUTIONS.items()
        }
        correction = RigidReferenceConfirmation(0, REFERENCES, CHECKPOINT)
        for xy in CALIBRATION:
            place(world, xy)
            truth = world.data.body("red_block").xpos.tolist()
            packets = {
                key: acquire(world, renderer, correction) for key, renderer in renderers.items()
            }
            report["calibration"].append(
                {"initial_xy": xy, "truth_xyz": truth, "profiles": packets}
            )
            save()
        fitted = {}
        for key in RESOLUTIONS:
            samples = [
                {"truth_xyz": row["truth_xyz"], **row["profiles"][key]}
                for row in report["calibration"]
            ]
            fitted[key] = estimate_profile_bounds(samples, min_pairs=4, min_pair_fraction=2 / 3)
        profiles = {key: value["position_error_m"] for key, value in fitted.items()}
        report["fitted_profiles"] = fitted
        frozen = json.dumps(profiles, sort_keys=True)
        report["frozen_profiles_sha256"] = hashlib.sha256(frozen.encode()).hexdigest()
        # Persist calibration before rendering any held-out position.
        save()
        print("frozen budgets", profiles, flush=True)
        for xy in VALIDATION:
            place(world, xy)
            start = float(world.data.time)
            checks = {
                "coarse": UncertaintyViewConfirmation(start, {"coarse": profiles["coarse"]}),
                "detail": UncertaintyViewConfirmation(
                    start, {"detail": profiles["detail"]}, initial="detail"
                ),
                "active": UncertaintyViewConfirmation(start, profiles),
                "combined": UncertaintyViewConfirmation(start, profiles, switch_on_missing=True),
            }
            first_truth = dict.fromkeys(checks)
            rows = []
            for tick in range(16):
                timestamp = start + tick * 0.2
                before = world.data.qpos.copy()
                packets = {
                    key: acquire(world, renderer, correction) for key, renderer in renderers.items()
                }
                if not np.array_equal(before, world.data.qpos):
                    raise RuntimeError("acquisition changed physical state")
                truth = world.data.body("red_block").xpos.tolist()
                for name, check in checks.items():
                    check.step(
                        timestamp,
                        lambda profile, packets=packets: (
                            packets[profile]["primary"],
                            packets[profile]["secondary"],
                        ),
                    )
                    if first_truth[name] is None and check.status == "confirmed_in_tray":
                        first_truth[name] = inside(truth)
                rows.append(
                    {
                        "time": timestamp,
                        "truth_xyz": truth,
                        "truth_inside": inside(truth),
                        "profiles": packets,
                    }
                )
                if tick < 15:
                    for _ in range(round(0.2 / world.model.opt.timestep)):
                        mujoco.mj_step(world.model, world.data)
            case = {
                "initial_xy": xy,
                "policies": {key: check.result() for key, check in checks.items()},
                "false_confirmations": {key: value is False for key, value in first_truth.items()},
                "samples": rows,
            }
            report["cases"].append(case)
            save()
            print(xy, {key: check.status for key, check in checks.items()}, flush=True)
        report["summary"] = {
            key: {
                "confirmations": sum(
                    c["policies"][key]["status"] == "confirmed_in_tray" for c in report["cases"]
                ),
                "false_confirmations": sum(c["false_confirmations"][key] for c in report["cases"]),
                "switches": sum(len(c["policies"][key]["switches"]) for c in report["cases"]),
            }
            for key in checks
        }
        report["validation_errors"] = {}
        for key, bound in profiles.items():
            maximum, missing, exceeded = np.zeros(3), 0, 0
            for case in report["cases"]:
                for row in case["samples"]:
                    for side in ("primary", "secondary"):
                        point = row["profiles"][key][side]
                        if point is None:
                            missing += 1
                        else:
                            error = np.abs(np.array(point) - row["truth_xyz"])
                            maximum = np.maximum(maximum, error)
                            exceeded += int(np.any(error > np.array(bound) + 1e-12))
            report["validation_errors"][key] = {
                "max_abs_error_m": maximum.tolist(),
                "missing_views": missing,
                "budget_exceeded_views": exceeded,
                "total_views": len(VALIDATION) * 16 * 2,
            }
        if frozen != json.dumps(profiles, sort_keys=True):
            raise RuntimeError("validation changed frozen profiles")
        if hashes != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources()}:
            raise RuntimeError("source changed during evaluation")
        report["active_improved"] = (
            report["summary"]["active"]["confirmations"]
            > report["summary"]["coarse"]["confirmations"]
        )
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()
        if world is not None:
            world.camera_renderer = None
            for renderer in renderers.values():
                renderer.close()
            world.close()
    print(
        json.dumps(
            {key: report[key] for key in ("summary", "validation_errors", "active_improved")},
            indent=2,
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
