"""Rendered sensor-corruption comparison with frozen reference coverage rules."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.perception import locate_red_cube
from embodied_agent.reference_perception import locate_references
from embodied_agent.rigid_reference import RigidReferenceConfirmation
from embodied_agent.uncertainty_view import UncertaintyViewConfirmation
from scripts.evaluate_rendered_precision import inside, place, sources
from scripts.evaluate_rigid_reference import CHECKPOINT, REFERENCES, make_world

FAULTS = (
    "nominal",
    "sparse_depth_stripes",
    "half_depth_missing",
    "depthless_distractor",
    "checkpoint_occlusion",
    "depth_noise_1mm",
)


def paired_capture(world, correction, fault, rng):
    packets = {name: {} for name in ("components", "coverage")}
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
        rgb, depth = rgb.copy(), depth.copy()
        red, green, blue = rgb.astype(float).transpose(2, 0, 1)
        white_pixels = (rgb.min(axis=2) > 100) & (rgb.max(axis=2) < 1.1 * rgb.min(axis=2))
        reference_pixels = (
            white_pixels
            | ((blue > 70) & (blue > 1.6 * green))
            | ((red > 70) & (green > 70) & (green > 1.6 * blue))
        )
        if fault in ("sparse_depth_stripes", "half_depth_missing"):
            period = 6 if fault == "sparse_depth_stripes" else 2
            columns = np.arange(depth.shape[1]) % period == 0
            depth[reference_pixels & columns[None, :]] = np.nan
        elif fault == "depthless_distractor":
            rgb[40:46, 40:46] = (0, 0, 255)
            depth[40:46, 40:46] = np.nan
        elif fault == "checkpoint_occlusion":
            yy, xx = np.nonzero(white_pixels)
            if len(yy):
                rgb[yy[yy >= np.median(yy)], xx[yy >= np.median(yy)]] = 0
        elif fault == "depth_noise_1mm":
            depth += rng.normal(0, 0.001, depth.shape)
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
            isolate = True
            refs = locate_references(
                rgb,
                depth,
                position,
                rotation,
                fovy,
                isolate_components=isolate,
                min_depth_fraction=0.8 if name == "coverage" else None,
            )
            diagnostic = {}
            check = locate_references(
                mask,
                depth,
                position,
                rotation,
                fovy,
                isolate_components=isolate,
                min_depth_fraction=0.8 if name == "coverage" else None,
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
    paths = sources() + [Path(__file__), budget_path]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "source_sha256": hashes,
        "budget_m": budget,
        "min_depth_fraction": 0.8,
        "seed": 20261105,
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
        for fi, fault in enumerate(FAULTS):
            for pi, xy in enumerate(((0.64, 0), (0.692, 0.003))):
                place(world, xy)
                rng = np.random.default_rng(20261105 + 100 * fi + pi)
                start = float(world.data.time)
                checks = {
                    k: UncertaintyViewConfirmation(start, {k: budget}, initial=k)
                    for k in ("components", "coverage")
                }
                first_truth = dict.fromkeys(checks)
                samples = []
                for tick in range(16):
                    timestamp = start + tick * 0.2
                    packets = paired_capture(world, correction, fault, rng)
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
                report["cases"].append(
                    {
                        "fault": fault,
                        "initial_xy": xy,
                        "policies": {k: c.result() for k, c in checks.items()},
                        "false_confirmations": {k: v is False for k, v in first_truth.items()},
                        "samples": samples,
                    }
                )
                save()
                print(fault, xy, {k: c.status for k, c in checks.items()}, flush=True)
        report["summary"] = {
            fault: {
                k: {
                    "confirmations": sum(
                        c["policies"][k]["status"] == "confirmed_in_tray"
                        for c in report["cases"]
                        if c["fault"] == fault
                    ),
                    "false_confirmations": sum(
                        c["false_confirmations"][k] for c in report["cases"] if c["fault"] == fault
                    ),
                    "usable_views": sum(
                        row["profiles"][k][side] is not None
                        for c in report["cases"]
                        if c["fault"] == fault
                        for row in c["samples"]
                        for side in ("primary", "secondary")
                    ),
                }
                for k in checks
            }
            for fault in FAULTS
        }
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
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
