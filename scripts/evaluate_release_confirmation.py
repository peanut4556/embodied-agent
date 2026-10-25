"""Standalone rendered landing audit; does not drive the guarded policy."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.physics import PhysicsWorld
from embodied_agent.release_confirmation import ReleaseConfirmation
from scripts.audit_contact_watchdog import advance

CASES = [
    ("center", [0.64, 0, 0.12], False),
    ("inner_left", [0.61, 0, 0.12], False),
    ("near_edge", [0.69, 0, 0.12], False),
    ("outside_x", [0.80, 0, 0.12], False),
    ("outside_y", [0.64, 0.16, 0.12], False),
    ("missing_observation", [0.64, 0, 0.12], True),
]


def cases_for_suite(suite):
    if suite == "baseline":
        return [(name, xyz, missing, [1, 0, 0, 0]) for name, xyz, missing in CASES]
    if suite != "expanded":
        raise ValueError("unknown landing suite")
    cases = [
        (f"grid-{i}-{j}", [x, y, 0.12], False, [1, 0, 0, 0])
        for i, x in enumerate((0.575, 0.61, 0.64, 0.675, 0.705))
        for j, y in enumerate((-0.055, 0, 0.055))
    ]
    for x in (0.64, 0.69):
        for angle in (15, 30, 45):
            half = np.deg2rad(angle) / 2
            cases.append(
                (
                    f"roll-{x}-{angle}",
                    [x, 0, 0.12],
                    False,
                    [float(np.cos(half)), float(np.sin(half)), 0, 0],
                )
            )
    cases.append(("missing_observation", [0.64, 0, 0.12], True, [1, 0, 0, 0]))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--view", choices=("original", "offset"), default="original")
    parser.add_argument("--suite", choices=("baseline", "expanded"), default="baseline")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    paths = (
        [Path(__file__)]
        + [
            Path("src/embodied_agent") / name
            for name in (
                "release_confirmation.py",
                "perception.py",
                "physics.py",
                "assets/tabletop.xml",
            )
        ]
        + [Path("scripts/audit_contact_watchdog.py")]
    )
    hashes = {str(p): digest(p) for p in paths}
    report = {
        "status": "running",
        "view": args.view,
        "camera_position": [0.64, 0.35, 1.2] if args.view == "offset" else [0.4, 0, 1.2],
        "scope": "Synthetic free-drop scenes, fixed arm; not end-to-end policy validation.",
        "suite": args.suite,
        "source_sha256": hashes,
        "cases": [],
    }
    for name, position, missing, quaternion in cases_for_suite(args.suite):
        world = PhysicsWorld(perception="rgbd", perception_size_mode="oriented")
        try:
            if args.view == "offset":
                world.model.camera("perception").pos[:] = [0.64, 0.35, 1.2]
            world.data.qpos[5:8] = position
            world.data.qpos[8:12] = quaternion
            world.data.qvel[:] = 0
            command = np.r_[world.data.qpos[:3], [0.04, 0.04]]
            world.data.ctrl[:] = command
            mujoco.mj_forward(world.model, world.data)
            check = ReleaseConfirmation(float(world.data.time))
            rows = []
            for i in range(17):
                point, error = None, None
                if not missing:
                    try:
                        point = world.locate_block().tolist()
                    except ValueError as exc:
                        error = str(exc)
                status = check.observe(float(world.data.time), point)
                rows.append(
                    {
                        "time": float(world.data.time),
                        "xyz": point,
                        "error": error,
                        "truth_xyz": world.data.body("red_block").xpos.tolist(),
                        "truth_quaternion": world.data.qpos[8:12].tolist(),
                        "truth_inside": world.inside_box(),
                        "status": status,
                    }
                )
                if status != "pending":
                    break
                for _ in range(5):
                    advance(world, command, 25)
            row = {
                "name": name,
                "initial_xyz": position,
                "initial_quaternion": quaternion,
                "missing_observation": missing,
                "status": check.status,
                "observations": rows,
            }
            report["cases"].append(row)
            print(name, check.status, rows[-1]["truth_inside"], flush=True)
        finally:
            world.close()
    assert hashes == {str(p): digest(p) for p in paths}, "sources changed during audit"
    report["false_confirmations"] = sum(
        c["status"] == "confirmed_in_tray" and not c["observations"][-1]["truth_inside"]
        for c in report["cases"]
    )
    report["status"] = "complete"
    report["outcomes"] = {
        status: sum(c["status"] == status for c in report["cases"])
        for status in ("confirmed_in_tray", "outside_tray", "unconfirmed")
    }
    report["unconfirmed_true_goals"] = sum(
        c["status"] != "confirmed_in_tray"
        and c["observations"][-1]["truth_inside"]
        and not c["missing_observation"]
        for c in report["cases"]
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["false_confirmations"] == 0, "false confirmation observed"
    center_name = "center" if args.suite == "baseline" else "grid-2-1"
    center = next(c for c in report["cases"] if c["name"] == center_name)
    assert center["status"] == "confirmed_in_tray", "center did not confirm"
    assert report["cases"][-1]["status"] == "unconfirmed", "missing evidence confirmed"


if __name__ == "__main__":
    main()
