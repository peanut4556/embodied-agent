"""Truth-only support/kinematic audit; virtual poses never drive recovery control."""

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "outputs/hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import mujoco
import numpy as np

from embodied_agent.correction_data import payload_digest
from embodied_agent.memory_policy import digest
from embodied_agent.physics import PhysicsWorld


def contacts(model, data, geom_ids):
    rows = []
    for i, c in enumerate(data.contact):
        if c.geom1 not in geom_ids and c.geom2 not in geom_ids:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        rows.append(
            {
                "geoms": [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom-{g}"
                    for g in (c.geom1, c.geom2)
                ],
                "distance_m": float(c.dist),
                "position": c.pos.tolist(),
                "contact_frame": c.frame.reshape(3, 3).tolist(),
                "force_contact_frame": force.tolist(),
                "constraint_active": bool(c.efc_address >= 0),
            }
        )
    return rows


def probe(world, x, z):
    """Independent static geometry query; original simulation state remains untouched."""
    model = world.model
    data = mujoco.MjData(model)
    data.qpos[:] = world.data.qpos
    try:
        command = world.ik(x, z)
    except ValueError as exc:
        return {"target_xz": [x, z], "reachable": False, "reason": str(exc)}
    within = bool(
        np.all(command >= model.actuator_ctrlrange[:3, 0])
        and np.all(command <= model.actuator_ctrlrange[:3, 1])
    )
    data.qpos[:5] = np.r_[command, 0.04, 0.04]
    data.ctrl[:] = data.qpos[:5]
    mujoco.mj_forward(model, data)
    bodies = {
        model.body(n).id for n in ("upper", "forearm", "wrist", "finger_left", "finger_right")
    }
    arm = {g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies}
    collisions = []
    for c in data.contact:
        if (int(c.geom1) in arm) != (int(c.geom2) in arm) and c.dist < -1e-5:
            collisions.append(
                {
                    "geoms": [
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom-{g}"
                        for g in (c.geom1, c.geom2)
                    ],
                    "penetration_m": float(-c.dist),
                }
            )
    upper_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body("upper").id}
    mount_names = {"pedestal"} | {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom-{g}" for g in upper_geoms
    }
    non_mount = [c for c in collisions if set(c["geoms"]) != mount_names]
    tip = data.site("tip").xpos.copy()
    return {
        "target_xz": [x, z],
        "reachable": within,
        "joints": command.tolist(),
        "tip_xyz": tip.tolist(),
        "fk_error_m": float(np.linalg.norm(tip[[0, 2]] - [x, z])),
        "collisions": collisions,
        "non_mount_collisions": non_mount,
    }


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path("outputs/datasets/corrections-contact-oriented-v1")
    output = Path("outputs/evaluations/support-reach-v1")
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "recording.json").read_text())
    source = payload_digest(root)
    if (
        manifest["status"] != "validated"
        or json.loads((root / "selection.json").read_text())["payload_sha256"] != source
    ):
        raise ValueError("changed or unvalidated source")
    if (
        manifest["model_sha256"] != digest("src/embodied_agent/assets/tabletop.xml")
        or manifest["fps"] != 25
    ):
        raise ValueError("physics or clock changed")
    record = manifest["episodes"][5]
    if record["takeover"]["tick"] != 84 or record["scenario"]["x"] != 0.302 or record["success"]:
        raise ValueError("unexpected failure fixture")
    dataset = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav")
    exact = dataset.hf_dataset.data.column("replay.action")
    offset = sum(e["frames"] for e in manifest["episodes"][:5])
    world = PhysicsWorld()
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "source_sha256": source,
        "script_sha256": digest(__file__),
        "model_sha256": digest("src/embodied_agent/assets/tabletop.xml"),
        "group": "train",
        "training_performed": False,
        "test_executed": False,
        "support": [],
    }
    try:
        for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
            getattr(world.data, key)[:] = record["initial_state"][key]
        world.data.time = record["initial_state"]["time"]
        for tick in range(159):
            mujoco.mj_forward(world.model, world.data)
            if not np.allclose(
                world.data.qpos[:5],
                dataset[offset + tick]["observation.state"].numpy(),
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError("replay state mismatch")
            if tick == 158:
                break
            world.data.ctrl[:] = exact[offset + tick].as_py()
            for _ in range(20):
                mujoco.mj_step(world.model, world.data)
        held = world.data.ctrl.copy()
        cube = world.model.geom("red_block").id
        for tick in range(101):
            mujoco.mj_forward(world.model, world.data)
            if not np.array_equal(held, world.data.ctrl):
                raise ValueError("audit changed hold command")
            if tick in (0, 100):
                report["support"].append(
                    {
                        "wait_seconds": tick / 25,
                        "xyz": world.data.body("red_block").xpos.tolist(),
                        "velocity": world.data.qvel[5:11].tolist(),
                        "contacts": contacts(world.model, world.data, {cube}),
                        "mass_kg": float(world.model.body_mass[world.model.body("red_block").id]),
                        "geom_friction": world.model.geom_friction[cube].tolist(),
                    }
                )
            if tick < 100:
                for _ in range(20):
                    mujoco.mj_step(world.model, world.data)
        xyz = world.data.body("red_block").xpos.copy()
        rotation = world.data.body("red_block").xmat.reshape(3, 3)
        corners = (
            np.array(
                [
                    [a, b, c]
                    for a in (-0.025, 0.025)
                    for b in (-0.025, 0.025)
                    for c in (-0.025, 0.025)
                ]
            )
            @ rotation.T
            + xyz
        )
        jac = np.zeros((3, world.model.nv))
        jacrot = np.zeros_like(jac)
        mujoco.mj_jacSite(world.model, world.data, jac, jacrot, world.model.site("tip").id)
        report["geometry"] = {
            "cube_aabb_min": corners.min(0).tolist(),
            "cube_aabb_max": corners.max(0).tolist(),
            "tip_jacobian_arm": jac[:, :3].tolist(),
            "tip_y": float(world.data.site("tip").xpos[1]),
            "existing_pick_y_limit_m": 0.015,
            "center_outside_existing_pick_plane": bool(abs(xyz[1]) > 0.015),
            "open_finger_inner_surfaces_y": [-0.052, 0.052],
            "independent_finger_gap_midpoint_range_m": [-0.02, 0.02],
        }
        original = world.data.qpos.copy()
        report["paths"] = []
        for dx in (-0.05, 0.0, 0.05):
            # High horizontal approach, then vertical descent, in independent scratch states.
            targets = [(float(x), 0.30) for x in np.linspace(0.22, xyz[0] + dx, 21)] + [
                (float(xyz[0] + dx), float(z)) for z in np.linspace(0.30, 0.03, 28)[1:]
            ]
            probes = [probe(world, x, z) for x, z in targets]
            report["paths"].append(
                {
                    "x_offset_m": dx,
                    "samples": len(probes),
                    "all_within_joint_bounds": all(p["reachable"] for p in probes),
                    "first_non_mount_collision": next(
                        (
                            {"index": i, **p}
                            for i, p in enumerate(probes)
                            if p.get("non_mount_collisions")
                        ),
                        None,
                    ),
                    "end_pose": probes[-1],
                    "non_mount_collision_free_samples": sum(
                        p["reachable"] and not p.get("non_mount_collisions") for p in probes
                    ),
                }
            )
        if not np.array_equal(world.data.qpos, original):
            raise ValueError("static probes altered source state")
        if payload_digest(root) != source:
            raise ValueError("source changed")
        report["status"] = "complete"
        report["limitations"] = [
            "simulation truth used only for geometric feasibility audit, never recovery commands",
            "sampled static paths do not prove swept-volume clearance, dynamic stability or controllability",
            "no posture recovery action executed, no new successful sample",
            "single fixed training failure; current RGB-D detector still rejects it",
        ]
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        world.close()
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/evaluations/support-reach-v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
