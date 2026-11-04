"""Rendered rigid-alignment comparison with a withheld elevated checkpoint."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.perception import locate_red_cube
from embodied_agent.physics import PhysicsWorld
from embodied_agent.reference_confirmation import ReferenceConfirmation
from embodied_agent.reference_perception import locate_references
from embodied_agent.rigid_reference import RigidReferenceConfirmation
from scripts.evaluate_reference_confirmation import COLORS, REFERENCES

CHECKPOINT = (0.48, 0.06, 0.034)
CENTER = np.mean(list(REFERENCES.values()), axis=0)
FAULTS = (
    "nominal",
    "shared_rotation",
    "primary_rotation",
    "scale_z",
    "checkpoint_missing",
    "checkpoint_moved",
    "wrong_catalog",
)


def make_world(fault):
    world = PhysicsWorld(perception="rgbd", perception_size_mode="oriented")
    root = ET.parse("src/embodied_agent/assets/tabletop.xml").getroot()
    for key, point in {**REFERENCES, "checkpoint": CHECKPOINT}.items():
        x, y, z = point
        if key == "checkpoint" and fault == "checkpoint_moved":
            x += 0.012
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            {
                "name": f"reference_{key}",
                "type": "box",
                "pos": f"{x} {y} {z - 0.002}",
                "size": "0.012 0.012 0.002",
                "rgba": COLORS.get(key, "1 1 1 1"),
                "contype": "0",
                "conaffinity": "0",
            },
        )
    world.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    world.data = mujoco.MjData(world.model)
    world.reset()
    if fault == "checkpoint_missing":
        world.model.geom("reference_checkpoint").rgba[3] = 0
    return world


def capture(world, side, fault):
    world.model.camera("perception").pos[:] = [0.64, side * 0.35, 1.2]
    mujoco.mj_camlight(world.model, world.data)
    if world.camera_renderer is None:
        world.camera_renderer = mujoco.Renderer(world.model, height=480, width=640)
    renderer = world.camera_renderer
    renderer.update_scene(world.data, camera="perception")
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    try:
        depth = renderer.render()
    finally:
        renderer.disable_depth_rendering()
    transform, shift = np.eye(3), np.zeros(3)
    if fault == "shared_rotation" or (fault == "primary_rotation" and side == 1):
        angle = np.deg2rad(2)
        transform = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        shift[0] = -0.02
    elif fault == "scale_z":
        transform[2, 2] = 2
    elif fault == "wrong_catalog":
        shift[0] = -0.02
    camera = world.model.camera("perception").id
    position = transform @ (world.data.cam_xpos[camera] - CENTER) + CENTER + shift
    rotation = transform @ world.data.cam_xmat[camera].reshape(3, 3)
    fovy = world.model.cam_fovy[camera]
    references = locate_references(rgb, depth, position, rotation, fovy)
    # Reuse the same finite-depth / clipping / compactness checks for a white patch.
    white = (rgb.min(axis=2) > 100) & (rgb.max(axis=2) < 1.1 * rgb.min(axis=2))
    mask_image = np.zeros_like(rgb)
    mask_image[white] = (0, 0, 255)
    detected = locate_references(mask_image, depth, position, rotation, fovy)
    if "blue" in detected:
        references["checkpoint"] = detected["blue"]
    try:
        target = locate_red_cube(rgb, depth, position, rotation, fovy, size_mode="oriented")["xyz"]
    except ValueError:
        target = None
    return target, references


def run_case(x, fault):
    world = make_world(fault)
    try:
        world.data.qpos[5:8] = [x, 0, 0.12]
        world.data.qpos[8:12] = [1, 0, 0, 0]
        world.data.qvel[:] = 0
        mujoco.mj_forward(world.model, world.data)
        delta = np.array((-0.02, 0, 0)) if fault == "wrong_catalog" else np.zeros(3)
        catalog = {key: np.array(point) + delta for key, point in REFERENCES.items()}
        start = float(world.data.time)
        checks = {
            "translation": ReferenceConfirmation(start, catalog),
            "rigid": RigidReferenceConfirmation(start, catalog, np.array(CHECKPOINT) + delta),
        }
        first_truth = dict.fromkeys(checks)
        rows = []
        for tick in range(16):
            timestamp = float(world.data.time)
            a, b = capture(world, 1, fault), capture(world, -1, fault)
            for name, check in checks.items():
                refs = [
                    packet[1]
                    if name == "rigid"
                    else {key: value for key, value in packet[1].items() if key != "checkpoint"}
                    for packet in (a, b)
                ]
                check.step(timestamp, a[0], b[0], *refs)
                if first_truth[name] is None and check.status == "confirmed_in_tray":
                    first_truth[name] = world.inside_box()
            rows.append(
                {
                    "time": timestamp,
                    "primary": a,
                    "secondary": b,
                    "truth_xyz": world.data.body("red_block").xpos.tolist(),
                    "truth_inside": world.inside_box(),
                }
            )
            if tick < 15:
                for _ in range(round(0.2 / world.model.opt.timestep)):
                    mujoco.mj_step(world.model, world.data)
        return {
            "initial_x": x,
            "fault": fault,
            **{name: check.result() for name, check in checks.items()},
            "false_confirmations": {key: value is False for key, value in first_truth.items()},
            "samples": rows,
        }
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path("scripts/evaluate_reference_confirmation.py")] + [
        Path("src/embodied_agent") / name
        for name in (
            "rigid_reference.py",
            "reference_confirmation.py",
            "reference_perception.py",
            "cross_view.py",
            "release_confirmation.py",
            "physics.py",
            "perception.py",
            "assets/tabletop.xml",
        )
    ]
    hashes = {str(p): digest(p) for p in paths}
    report = {
        "schema_version": 1,
        "status": "running",
        "source_sha256": hashes,
        "scope": "Rendered synchronized free-drop experiment. Three fitting markers and one withheld raised marker. "
        "2 degree rotations with -20 mm x shift; scale_z is a deliberately nonrigid coordinate transform, not a camera rotation.",
        "cases": [],
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for fault in FAULTS:
            for x in (0.64, 0.705):
                row = run_case(x, fault)
                report["cases"].append(row)
                save()
                print(
                    fault,
                    x,
                    row["translation"]["status"],
                    row["rigid"]["status"],
                    row["false_confirmations"],
                    flush=True,
                )
        report["summary"] = {
            fault: {
                name: {
                    "confirmations": sum(
                        row[name]["status"] == "confirmed_in_tray"
                        for row in report["cases"]
                        if row["fault"] == fault
                    ),
                    "false_confirmations": sum(
                        row["false_confirmations"][name]
                        for row in report["cases"]
                        if row["fault"] == fault
                    ),
                }
                for name in ("translation", "rigid")
            }
            for fault in FAULTS
        }
        if hashes != {str(p): digest(p) for p in paths}:
            raise RuntimeError("source changed during evaluation")
        summary = report["summary"]
        report["matches_expected"] = (
            all(
                summary[f]["rigid"] == {"confirmations": 1, "false_confirmations": 0}
                for f in ("nominal", "shared_rotation", "primary_rotation")
            )
            and all(
                summary[f]["rigid"]["confirmations"] == 0
                for f in ("scale_z", "checkpoint_missing", "checkpoint_moved")
            )
            and summary["wrong_catalog"]["rigid"]["false_confirmations"] == 1
        )
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()
    return int(not report["matches_expected"])


if __name__ == "__main__":
    raise SystemExit(main())
