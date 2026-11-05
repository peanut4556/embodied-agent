"""Same-image component isolation, fixed budgets and frozen-rule held-out replay."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.perception import locate_red_cube
from embodied_agent.profile_admission import QualifiedViewConfirmation, admit_profiles
from embodied_agent.reference_perception import locate_references
from embodied_agent.rigid_reference import RigidReferenceConfirmation
from embodied_agent.uncertainty_view import UncertaintyViewConfirmation
from scripts.evaluate_profile_admission import SCENES
from scripts.evaluate_rendered_precision import inside, place, sources
from scripts.evaluate_rigid_reference import CHECKPOINT, REFERENCES, make_world

HOLDOUT = [
    (0.596, -0.004, 0.10),
    (0.604, 0.012, -0.20),
    (0.618, -0.025, 0.20),
    (0.654, 0.024, -0.10),
    (0.681, -0.012, 0.20),
    (0.686, 0.005, 0.10),
    (0.638, -0.036, -0.20),
    (0.645, 0.036, 0.10),
    (0.587, -0.003, 0),
    (0.693, 0.003, 0),
    (0.635, -0.044, 0),
    (0.648, 0.044, 0),
]


def paired_capture(world, correction):
    packets = {name: {} for name in ("legacy", "components")}
    for side, label in ((1, "primary"), (-1, "secondary")):
        world.model.camera("perception").pos[:] = [0.64, side * 0.35, 1.2]
        mujoco.mj_camlight(world.model, world.data)
        renderer = world.camera_renderer
        renderer.update_scene(world.data, camera="perception")
        rgb = renderer.render()
        renderer.enable_depth_rendering()
        try:
            depth = renderer.render()
        finally:
            renderer.disable_depth_rendering()
        camera = world.model.camera("perception").id
        position = world.data.cam_xpos[camera]
        rotation = world.data.cam_xmat[camera].reshape(3, 3)
        fovy = world.model.cam_fovy[camera]
        try:
            target = locate_red_cube(rgb, depth, position, rotation, fovy, size_mode="oriented")[
                "xyz"
            ]
        except ValueError:
            target = None
        white = (rgb.min(axis=2) > 100) & (rgb.max(axis=2) < 1.1 * rgb.min(axis=2))
        mask = np.zeros_like(rgb)
        mask[white] = (0, 0, 255)
        for name, packet in packets.items():
            isolate = name == "components"
            refs = locate_references(
                rgb, depth, position, rotation, fovy, isolate_components=isolate
            )
            diagnostic = {}
            check = locate_references(
                mask,
                depth,
                position,
                rotation,
                fovy,
                isolate_components=isolate,
                diagnostics=diagnostic,
            )
            if "blue" in check:
                refs["checkpoint"] = check["blue"]
            corrected, decision = correction.correct(target, refs)
            packet[label] = corrected
            packet[label + "_correction"] = decision
            packet[label + "_checkpoint_components"] = diagnostic
    return packets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budget_path = Path("docs/evaluations/rendered-precision-v3.json")
    budget = json.loads(budget_path.read_text())["fitted_profiles"]["detail"]["position_error_m"]
    budgets = {key: budget for key in ("legacy", "components")}
    paths = sources() + [
        budget_path,
        Path("scripts/evaluate_reference_components.py"),
        Path("scripts/evaluate_profile_admission.py"),
        Path("src/embodied_agent/profile_admission.py"),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "source_sha256": hashes,
        "frozen_budget_m": budget,
        "development_scenes": SCENES,
        "holdout_scenes": HOLDOUT,
        "cases": [],
    }

    def save():
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    save()
    world = None
    try:
        world = make_world("nominal")
        world.camera_renderer = mujoco.Renderer(world.model, height=480, width=640)
        correction = RigidReferenceConfirmation(0, REFERENCES, CHECKPOINT)
        development = []
        for phase, scenes in (("development", SCENES), ("holdout", HOLDOUT)):
            if phase == "holdout":
                admission = admit_profiles(budgets, ["original_calibration"], development)
                report["frozen_admission"] = admission.result()
                save()
                print("frozen admission", dict(admission.profiles), flush=True)
            for index, (x, y, yaw) in enumerate(scenes):
                place(world, (x, y))
                world.data.qpos[8:12] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                world.data.qvel[:] = 0
                mujoco.mj_forward(world.model, world.data)
                for _ in range(500):
                    mujoco.mj_step(world.model, world.data)
                start = float(world.data.time)
                checks = {
                    key: UncertaintyViewConfirmation(start, {key: budget}, initial=key)
                    for key in budgets
                }
                if phase == "holdout":
                    checks["qualified"] = QualifiedViewConfirmation(
                        start, admission, initial="components"
                    )
                first_truth = dict.fromkeys(checks)
                samples = []
                for tick in range(16):
                    timestamp = start + tick * 0.2
                    before = world.data.qpos.copy()
                    packets = paired_capture(world, correction)
                    if not np.array_equal(before, world.data.qpos):
                        raise RuntimeError("capture altered physical state")
                    truth = world.data.body("red_block").xpos.tolist()
                    for key, check in checks.items():
                        check.step(
                            timestamp,
                            lambda profile, packets=packets: (
                                packets[profile]["primary"],
                                packets[profile]["secondary"],
                            ),
                        )
                        if first_truth[key] is None and check.status == "confirmed_in_tray":
                            first_truth[key] = inside(truth)
                    samples.append({"time": timestamp, "truth_xyz": truth, "profiles": packets})
                    if tick < 15:
                        for _ in range(round(0.2 / world.model.opt.timestep)):
                            mujoco.mj_step(world.model, world.data)
                case = {
                    "id": f"{phase}_{index}",
                    "phase": phase,
                    "scene": (x, y, yaw),
                    "policies": {key: check.result() for key, check in checks.items()},
                    "false_confirmations": {
                        key: value is False for key, value in first_truth.items()
                    },
                    "samples": samples,
                }
                report["cases"].append(case)
                if phase == "development":
                    development.append({"id": case["id"], "samples": samples})
                save()
                print(
                    phase,
                    (x, y, yaw),
                    {key: check.status for key, check in checks.items()},
                    flush=True,
                )
        report["summary"] = {}
        for phase in ("development", "holdout"):
            rows = [c for c in report["cases"] if c["phase"] == phase]
            report["summary"][phase] = {
                key: {
                    "confirmations": sum(
                        c["policies"][key]["status"] == "confirmed_in_tray" for c in rows
                    ),
                    "false_confirmations": sum(c["false_confirmations"][key] for c in rows),
                }
                for key in rows[0]["policies"]
            }
            report[phase + "_audit"] = admit_profiles(
                budgets,
                ["original_calibration"],
                [{"id": c["id"], "samples": c["samples"]} for c in rows],
            ).result()
        if hashes != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:
            raise RuntimeError("source changed during evaluation")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()
        if world is not None:
            world.close()
    print(
        json.dumps(
            {key: report[key] for key in ("summary", "development_audit", "holdout_audit")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
