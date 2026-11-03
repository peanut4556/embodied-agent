"""Rendered fixed references versus shared camera translation; no target truth input."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.cross_view import CrossViewConfirmation
from embodied_agent.memory_policy import digest
from embodied_agent.perception import locate_red_cube
from embodied_agent.physics import PhysicsWorld
from embodied_agent.reference_confirmation import ReferenceConfirmation
from embodied_agent.reference_perception import locate_references

REFERENCES = {
    "blue": (0.48, -0.12, 0.004),
    "yellow": (0.64, -0.12, 0.004),
    "magenta": (0.80, 0.12, 0.004),
}
COLORS = {"blue": "0.05 0.05 1 1", "yellow": "1 1 0.05 1", "magenta": "1 0.05 1 1"}
FAULTS = (
    "nominal",
    "shared_bias",
    "primary_bias",
    "missing_reference",
    "inconsistent_reference",
    "excessive_bias",
    "wrong_catalog",
)


def make_world(fault):
    world = PhysicsWorld(perception="rgbd", perception_size_mode="oriented")
    root = ET.parse("src/embodied_agent/assets/tabletop.xml").getroot()
    for key, point in REFERENCES.items():
        x, y, z = point
        if fault == "inconsistent_reference" and key == "blue":
            x += 0.012
        ET.SubElement(
            root.find("worldbody"),
            "geom",
            {
                "name": f"reference_{key}",
                "type": "box",
                "pos": f"{x} {y} {z - 0.002}",
                "size": "0.014 0.014 0.002",
                "rgba": COLORS[key],
                "contype": "0",
                "conaffinity": "0",
            },
        )
    world.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    world.data = mujoco.MjData(world.model)
    world.reset()
    if fault == "missing_reference":
        world.model.geom("reference_blue").rgba[3] = 0
    return world


def capture(world, side, bias):
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
    camera = world.model.camera("perception").id
    position = world.data.cam_xpos[camera] + np.array([bias, 0, 0])
    rotation, fovy = world.data.cam_xmat[camera], world.model.cam_fovy[camera]
    references = locate_references(rgb, depth, position, rotation, fovy)
    try:
        target = locate_red_cube(rgb, depth, position, rotation, fovy, size_mode="oriented")["xyz"]
        error = None
    except ValueError as exc:
        target, error = None, str(exc)
    return target, references, error


def run_case(x, fault):
    world = make_world(fault)
    try:
        world.data.qpos[5:8] = [x, 0, 0.12]
        world.data.qpos[8:12] = [1, 0, 0, 0]
        world.data.qvel[:] = 0
        mujoco.mj_forward(world.model, world.data)
        catalog = {
            key: (p[0] - (0.02 if fault == "wrong_catalog" else 0), p[1], p[2])
            for key, p in REFERENCES.items()
        }
        start = float(world.data.time)
        raw, anchored = CrossViewConfirmation(start), ReferenceConfirmation(start, catalog)
        rows, first_truth = [], {"raw": None, "anchored": None}
        for tick in range(16):
            timestamp = float(world.data.time)
            captures = []
            for index, side in enumerate((1, -1)):
                bias = (
                    -0.06
                    if fault == "excessive_bias"
                    else -0.02
                    if fault in {"shared_bias", "wrong_catalog"}
                    or (fault == "primary_bias" and index == 0)
                    else 0
                )
                captures.append(capture(world, side, bias))
            primary, secondary = captures
            raw.step(timestamp, primary[0], secondary[0])
            anchored.step(timestamp, primary[0], secondary[0], primary[1], secondary[1])
            truth = world.inside_box()
            for name, check in (("raw", raw), ("anchored", anchored)):
                if first_truth[name] is None and check.status == "confirmed_in_tray":
                    first_truth[name] = truth
            rows.append(
                {
                    "time": timestamp,
                    "primary": primary,
                    "secondary": secondary,
                    "truth_inside": truth,
                    "truth_xyz": world.data.body("red_block").xpos.tolist(),
                }
            )
            if tick < 15:
                for _ in range(round(0.2 / world.model.opt.timestep)):
                    mujoco.mj_step(world.model, world.data)
        return {
            "initial_x": x,
            "fault": fault,
            "raw": raw.result(),
            "anchored": anchored.result(),
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
    paths = [Path(__file__)] + [
        Path("src/embodied_agent") / name
        for name in (
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
        "scope": "Synchronous simulated RGB-D captures, three identified fixed visual references. "
        "Reference catalog is a supplied design prior, not a live target truth measurement.",
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
                    row["raw"]["status"],
                    row["anchored"]["status"],
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
                for name in ("raw", "anchored")
            }
            for fault in FAULTS
        }
        if hashes != {str(p): digest(p) for p in paths}:
            raise RuntimeError("source changed during evaluation")
        summary = report["summary"]
        report["matches_expected"] = (
            all(
                summary[f]["anchored"] == {"confirmations": 1, "false_confirmations": 0}
                for f in ("nominal", "shared_bias", "primary_bias")
            )
            and all(
                summary[f]["anchored"]["confirmations"] == 0
                for f in ("missing_reference", "inconsistent_reference", "excessive_bias")
            )
            and summary["shared_bias"]["raw"]["false_confirmations"] == 1
            and summary["wrong_catalog"]["anchored"]["false_confirmations"] == 1
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
