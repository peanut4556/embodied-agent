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
    if suite == "validation":
        cases = [("center", [0.64, 0, 0.12], False, [1, 0, 0, 0])]
        for x in (0.60, 0.655, 0.685):
            for angle in (20, 35, 50, 65):
                half = np.deg2rad(angle) / 2
                cases.append(
                    (
                        f"new-roll-{x}-{angle}",
                        [x, -0.01, 0.12],
                        False,
                        [float(np.cos(half)), float(np.sin(half)), 0, 0],
                    )
                )
        for x in (0.57, 0.72):
            for y in (-0.065, 0.065):
                cases.append((f"outside-{x}-{y}", [x, y, 0.12], False, [1, 0, 0, 0]))
        for mode in ("early", "intermittent", True):
            cases.append((f"missing-{mode}", [0.64, 0, 0.12], mode, [1, 0, 0, 0]))
        return cases
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


def observation_missing(mode, sample):
    if type(mode) is bool:
        return mode
    if mode == "early":
        return sample < 18  # First 3.6 seconds at 5 Hz.
    if mode == "intermittent":
        return sample % 2 == 0
    raise ValueError("unknown missing-observation schedule")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--view", choices=("original", "offset"), default="original")
    parser.add_argument(
        "--suite", choices=("baseline", "expanded", "validation"), default="baseline"
    )
    parser.add_argument("--observe-seconds", type=int, choices=(3, 6), default=3)
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
        "confirmation_schema_version": 2,
        "view": args.view,
        "camera_position": [0.64, 0.35, 1.2] if args.view == "offset" else [0.4, 0, 1.2],
        "scope": "Synthetic free-drop scenes, fixed arm; not end-to-end policy validation.",
        "suite": args.suite,
        "observe_seconds": args.observe_seconds,
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
            late_check = None
            rows = []
            for i in range(args.observe_seconds * 5 + 1):
                point, error = None, None
                if not observation_missing(missing, i):
                    try:
                        point = world.locate_block().tolist()
                    except ValueError as exc:
                        error = str(exc)
                status = check.observe(float(world.data.time), point)
                if args.observe_seconds == 6 and i >= 15:
                    if late_check is None:
                        late_check = ReleaseConfirmation(float(world.data.time))
                    late_check.observe(float(world.data.time), point)
                rows.append(
                    {
                        "time": float(world.data.time),
                        "xyz": point,
                        "error": error,
                        "truth_xyz": world.data.body("red_block").xpos.tolist(),
                        "truth_quaternion": world.data.qpos[8:12].tolist(),
                        "truth_inside": world.inside_box(),
                        "status": status,
                        "reason": check.reason,
                        "late_status": late_check.status if late_check else None,
                    }
                )
                if status != "pending" and args.observe_seconds == 3:
                    break
                for _ in range(5):
                    advance(world, command, 25)
            row = {
                "name": name,
                "initial_xyz": position,
                "initial_quaternion": quaternion,
                "missing_observation": missing,
                "status": check.status,
                "confirmation_result": check.result(),
                "late_result": late_check.result() if late_check else None,
                "observations": rows,
                "first_confirmation_elapsed_s": next(
                    (o["time"] - check.start for o in rows if o["status"] == "confirmed_in_tray"),
                    None,
                ),
                "late_confirmation_elapsed_s": next(
                    (
                        o["time"] - check.start
                        for o in rows
                        if o["late_status"] == "confirmed_in_tray"
                    ),
                    None,
                ),
            }
            report["cases"].append(row)
            print(name, check.status, late_check.status if late_check else None, flush=True)
        finally:
            world.close()
    assert hashes == {str(p): digest(p) for p in paths}, "sources changed during audit"
    report["false_confirmations"] = sum(
        c["status"] == "confirmed_in_tray"
        and not next(o for o in c["observations"] if o["status"] != "pending")["truth_inside"]
        for c in report["cases"]
    )
    report["status"] = "complete"
    if args.observe_seconds == 6:
        report["late_false_confirmations"] = sum(
            c["late_result"]["status"] == "confirmed_in_tray"
            and not next(o for o in c["observations"] if o["late_status"] == "confirmed_in_tray")[
                "truth_inside"
            ]
            for c in report["cases"]
        )
    report["outcomes"] = {
        status: sum(c["status"] == status for c in report["cases"])
        for status in ("confirmed_in_tray", "outside_confirmation_region", "unconfirmed")
    }
    report["unconfirmed_true_goals"] = sum(
        c["status"] != "confirmed_in_tray"
        and c["observations"][-1]["truth_inside"]
        and not c["missing_observation"]
        for c in report["cases"]
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["false_confirmations"] == 0, "false confirmation observed"
    assert report.get("late_false_confirmations", 0) == 0, "late false confirmation observed"
    center_name = "grid-2-1" if args.suite == "expanded" else "center"
    center = next(c for c in report["cases"] if c["name"] == center_name)
    assert center["status"] == "confirmed_in_tray", "center did not confirm"
    assert report["cases"][-1]["status"] == "unconfirmed", "missing evidence confirmed"


if __name__ == "__main__":
    main()
