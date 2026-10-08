"""Scratch-state gripper geometry and interval sweep screening; no action execution."""

import json
from itertools import pairwise
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.gripper_sweep import box_aabb, overlaps, swept_box
from embodied_agent.memory_policy import digest
from embodied_agent.physics import PhysicsWorld

GRIPPER = ("palm", "finger_left", "finger_right")


def gripper_boxes(model, data, q):
    if np.any(q < model.actuator_ctrlrange[:, 0]) or np.any(q > model.actuator_ctrlrange[:, 1]):
        raise ValueError("joint limit exceeded")
    data.qpos[:5] = q
    mujoco.mj_forward(model, data)
    return {
        name: box_aabb(data.geom(name).xpos, data.geom(name).xmat, model.geom(name).size)
        for name in GRIPPER
    }


def main():
    support = json.loads(Path("docs/evaluations/support-reach-v1.json").read_text())
    stress = json.loads(Path("docs/evaluations/envelope-stress-v1.json").read_text())
    selected = [
        r
        for r in stress["cases"]
        if r["episode"] == 5
        and r["camera_shift_x_m"] == 0
        and r["occlusion"] == "none"
        and r["depth_perturbation"] == "none"
    ]
    if len(selected) != 1:
        raise ValueError("ambiguous source bound")
    bound = selected[0]["estimate"]["bound"]
    world = PhysicsWorld()
    model = world.model
    if digest("src/embodied_agent/assets/tabletop.xml") != support["model_sha256"]:
        raise ValueError("model differs from support audit")
    data = mujoco.MjData(model)
    data.qpos[:] = world.data.qpos
    original = world.data.qpos.copy()
    # Triangle-inequality reach bound about each arm joint; finger extension included.
    wrist_reach = max(
        float(np.linalg.norm(model.geom("palm").size)),
        float(
            np.linalg.norm(model.body("finger_left").pos)
            + 0.04
            + np.linalg.norm(model.geom("finger_left").size)
        ),
        float(
            np.linalg.norm(model.body("finger_right").pos)
            + 0.04
            + np.linalg.norm(model.geom("finger_right").size)
        ),
    )
    fore = float(np.linalg.norm(model.body("wrist").pos))
    upper = float(np.linalg.norm(model.body("forearm").pos))
    radii = np.array([upper + fore + wrist_reach, fore + wrist_reach, wrist_reach])
    obstacles = {"object_envelope": (bound["aabb_min"], bound["aabb_max"])}
    for name in ("table", "tray_floor", "tray_left", "tray_right", "tray_back", "tray_front"):
        obstacles[name] = box_aabb(
            world.data.geom(name).xpos, world.data.geom(name).xmat, model.geom(name).size
        )
    report = {
        "status": "complete",
        "lever_radii_m": radii.tolist(),
        "script_sha256": digest(__file__),
        "sweep_sha256": digest("src/embodied_agent/gripper_sweep.py"),
        "source_reports": {
            n: digest("docs/evaluations/" + n)
            for n in ("support-reach-v1.json", "envelope-stress-v1.json")
        },
        "paths": [],
        "training_performed": False,
        "test_executed": False,
        "clearance_certified": False,
    }
    try:
        center = support["support"][-1]["xyz"][0]
        for name, points in [("high_only", [(0.22, 0.30), (center, 0.30)])] + [
            (f"descend-{dx:+.2f}", [(0.22, 0.30), (center + dx, 0.30), (center + dx, 0.03)])
            for dx in (-0.05, 0, 0.05)
        ]:
            waypoints = []
            for a, b in pairwise(points):
                # 10 mm Cartesian waypoint spacing, interpolation between them is joint-linear.
                count = int(np.ceil(np.linalg.norm(np.array(b) - a) / 0.01))
                for fraction in np.linspace(0, 1, count + 1)[:-1]:
                    x, z = np.array(a) + (np.array(b) - a) * fraction
                    waypoints.append(np.r_[world.ik(x, z), 0.04, 0.04])
            waypoints.append(np.r_[world.ik(*points[-1]), 0.04, 0.04])
            boxes = [gripper_boxes(model, data, q) for q in waypoints]
            hits = []
            point_hits = []
            max_padding = 0.0
            for i, box in enumerate(boxes):
                for geom in GRIPPER:
                    for obstacle, volume in obstacles.items():
                        if overlaps(box[geom], volume):
                            point_hits.append({"waypoint": i, "geom": geom, "obstacle": obstacle})
            for i, (q0, q1) in enumerate(pairwise(waypoints)):
                for geom in GRIPPER:
                    sweep = swept_box(boxes[i][geom], boxes[i + 1][geom], q0, q1, radii)
                    max_padding = max(
                        max_padding,
                        float(
                            np.max(np.minimum(boxes[i][geom][0], boxes[i + 1][geom][0]) - sweep[0])
                        ),
                    )
                    # Dense samples check implementation; analytical length bound covers continuum.
                    for t in np.linspace(0, 1, 11):
                        actual = gripper_boxes(model, data, q0 + (q1 - q0) * t)[geom]
                        if np.any(actual[0] < sweep[0] - 1e-10) or np.any(
                            actual[1] > sweep[1] + 1e-10
                        ):
                            raise ValueError("sweep underbound")
                    for obstacle, volume in obstacles.items():
                        if overlaps(sweep, volume):
                            hits.append({"segment": i, "geom": geom, "obstacle": obstacle})
            report["paths"].append(
                {
                    "name": name,
                    "waypoints": len(waypoints),
                    "endpoint_overlaps": len(point_hits),
                    "sweep_overlaps": len(hits),
                    "first_point_overlap": point_hits[0] if point_hits else None,
                    "first_sweep_overlap": hits[0] if hits else None,
                    "obstacles_hit": sorted({h["obstacle"] for h in hits}),
                    "max_interval_padding_m": max_padding,
                    "gripper_screen": "blocked_or_uncertain"
                    if hits
                    else "no_overlap_with_supplied_volumes",
                }
            )
        if not np.array_equal(original, world.data.qpos):
            raise ValueError("audit changed live state")
    finally:
        world.close()
    report["limitations"] = [
        "palm/fingers only: excludes arm links, self-collision, base overlap and unobserved obstacles",
        "object bound conditional on size and sensing; tray geometry known from fixture, not estimated",
        "linear joint interpolation only; no rate-limited/dynamic execution tracking",
        "AABB overlap is conservative screening, not proof of physical collision",
        "no contact maneuver executed; no success-rate claim",
    ]
    Path("docs/evaluations/gripper-sweep-v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
