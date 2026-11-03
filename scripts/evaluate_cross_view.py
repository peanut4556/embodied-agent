"""Rendered paired-view calibration fault experiment; truth is scoring only."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.cross_view import CrossViewConfirmation
from embodied_agent.memory_policy import digest
from embodied_agent.perception import locate_red_cube
from embodied_agent.physics import PhysicsWorld
from embodied_agent.release_confirmation import ReleaseConfirmation


def read_view(world, side, bias):
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
    return locate_red_cube(
        rgb,
        depth,
        world.data.cam_xpos[camera] + np.array([bias, 0, 0]),
        world.data.cam_xmat[camera],
        world.model.cam_fovy[camera],
        size_mode="oriented",
    )["xyz"]


def run_case(x, fault):
    world = PhysicsWorld(perception="rgbd", perception_size_mode="oriented")
    try:
        world.data.qpos[5:8] = [x, 0, 0.12]
        world.data.qpos[8:12] = [1, 0, 0, 0]
        world.data.qvel[:] = 0
        mujoco.mj_forward(world.model, world.data)
        start = float(world.data.time)
        single, paired = ReleaseConfirmation(start), CrossViewConfirmation(start)
        rows = []
        single_truth = paired_truth = None
        for tick in range(16):
            timestamp = float(world.data.time)
            points, errors = [], []
            for index, side in enumerate((1, -1)):
                bias = (
                    -0.02
                    if fault == "shared_bias" or (fault == "primary_bias" and index == 0)
                    else 0
                )
                try:
                    point = (
                        None
                        if fault == "secondary_missing" and index == 1
                        else read_view(world, side, bias)
                    )
                    points.append(point)
                    errors.append(None)
                except ValueError as exc:
                    points.append(None)
                    errors.append(str(exc))
            single.observe(timestamp, points[0])
            paired.step(timestamp, *points)
            truth = world.inside_box()
            if single_truth is None and single.status == "confirmed_in_tray":
                single_truth = truth
            if paired_truth is None and paired.status == "confirmed_in_tray":
                paired_truth = truth
            rows.append(
                {
                    "time": timestamp,
                    "primary": points[0],
                    "secondary": points[1],
                    "errors": errors,
                    "truth_xyz": world.data.body("red_block").xpos.tolist(),
                    "truth_inside": truth,
                    "single_status": single.status,
                    "paired_status": paired.status,
                }
            )
            if tick < 15:
                for _ in range(round(0.2 / world.model.opt.timestep)):
                    mujoco.mj_step(world.model, world.data)
        return {
            "initial_x": x,
            "fault": fault,
            "single": single.result(),
            "paired": paired.result(),
            "single_false_confirmation": single_truth is False,
            "paired_false_confirmation": paired_truth is False,
            "final_truth_inside": world.inside_box(),
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
    sources = [Path(__file__)] + [
        Path("src/embodied_agent") / name
        for name in (
            "cross_view.py",
            "release_confirmation.py",
            "perception.py",
            "physics.py",
            "assets/tabletop.xml",
        )
    ]
    hashes = {str(p): digest(p) for p in sources}
    report = {
        "schema_version": 1,
        "status": "running",
        "source_sha256": hashes,
        "scope": "Fixed-arm simulated free drops; synchronized rendered views. "
        "Calibration fault is an x translation of -0.02 m, not object motion. "
        "Shared calibration bias is a deliberate undetectable counterexample.",
        "cases": [],
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for fault in ("nominal", "primary_bias", "secondary_missing", "shared_bias"):
            for x in (0.61, 0.64, 0.705, 0.8):
                row = run_case(x, fault)
                report["cases"].append(row)
                save()
                print(
                    fault,
                    x,
                    row["single"]["status"],
                    row["paired"]["status"],
                    row["single_false_confirmation"],
                    row["paired_false_confirmation"],
                    flush=True,
                )
        report["summary"] = {
            fault: {
                "single_confirmations": sum(
                    r["single"]["status"] == "confirmed_in_tray"
                    for r in report["cases"]
                    if r["fault"] == fault
                ),
                "paired_confirmations": sum(
                    r["paired"]["status"] == "confirmed_in_tray"
                    for r in report["cases"]
                    if r["fault"] == fault
                ),
                "single_false_confirmations": sum(
                    r["single_false_confirmation"] for r in report["cases"] if r["fault"] == fault
                ),
                "paired_false_confirmations": sum(
                    r["paired_false_confirmation"] for r in report["cases"] if r["fault"] == fault
                ),
            }
            for fault in ("nominal", "primary_bias", "secondary_missing", "shared_bias")
        }
        if hashes != {str(p): digest(p) for p in sources}:
            raise RuntimeError("source changed during evaluation")
        summary = report["summary"]
        report["passed"] = (
            summary["nominal"]["paired_confirmations"] == 2
            and summary["nominal"]["paired_false_confirmations"] == 0
            and summary["primary_bias"]["single_false_confirmations"] > 0
            and summary["primary_bias"]["paired_false_confirmations"] == 0
            and summary["secondary_missing"]["paired_confirmations"] == 0
            and summary["shared_bias"]["paired_false_confirmations"] > 0
        )
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
