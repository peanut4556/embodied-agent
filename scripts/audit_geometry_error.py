"""Score depth, red boundary and wall association errors at four fixed replay states."""

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "outputs/hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import mujoco
import numpy as np
from PIL import Image

from embodied_agent.correction_data import payload_digest
from embodied_agent.geometry_error import erode, ray_box_depth, stats, surface_distance
from embodied_agent.memory_policy import digest
from embodied_agent.perception import locate_red_cube
from embodied_agent.physics import PhysicsWorld
from embodied_agent.visible_geometry import point_cloud


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path("outputs/datasets/corrections-contact-stable-v1")
    output = Path("outputs/evaluations/geometry-error-v1")
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "recording.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    source = payload_digest(root)
    if manifest["status"] != "validated" or source != selection["payload_sha256"]:
        raise ValueError("validated unchanged dataset required")
    dataset = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav")
    exact = dataset.hf_dataset.data.column("replay.action")
    offsets = np.cumsum([0] + [e["frames"] for e in manifest["episodes"]])
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "source_sha256": source,
        "scoring_sha256": digest("src/embodied_agent/geometry_error.py"),
        "script_sha256": digest(__file__),
        "perception_sha256": digest("src/embodied_agent/perception.py"),
        "group": "train",
        "training_performed": False,
        "test_executed": False,
        "cases": [],
    }
    try:
        for ep, wait in ((3, 0), (4, 0), (5, 0), (5, 100)):
            record = manifest["episodes"][ep]
            tick = record["takeover"]["tick"] + 3 * manifest["fps"] - 1
            world = PhysicsWorld()
            renderer = None
            try:
                for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                    getattr(world.data, key)[:] = record["initial_state"][key]
                world.data.time = record["initial_state"]["time"]
                for t in range(tick + 1):
                    mujoco.mj_forward(world.model, world.data)
                    if not np.allclose(
                        world.data.qpos[:5],
                        dataset[int(offsets[ep]) + t]["observation.state"].numpy(),
                        atol=1e-6,
                        rtol=0,
                    ):
                        raise ValueError("replay state mismatch")
                    if t == tick:
                        break
                    world.data.ctrl[:] = exact[int(offsets[ep]) + t].as_py()
                    for _ in range(round(1 / 25 / world.model.opt.timestep)):
                        mujoco.mj_step(world.model, world.data)
                for _ in range(wait * round(1 / 25 / world.model.opt.timestep)):
                    mujoco.mj_step(world.model, world.data)
                mujoco.mj_forward(world.model, world.data)
                renderer = mujoco.Renderer(world.model, height=480, width=640)
                renderer.update_scene(world.data, camera="perception")
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                camera = world.model.camera("perception").id
                pos, rot = world.data.cam_xpos[camera].copy(), world.data.cam_xmat[camera].copy()
                diagnostic = {}
                try:
                    detection = locate_red_cube(
                        rgb, depth, pos, rot, world.model.cam_fovy[camera], diagnostic
                    )
                    error = None
                except ValueError as exc:
                    detection, error = None, str(exc)
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                cube = (segmentation[:, :, 0] == world.model.geom("red_block").id) & (
                    segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
                )
                red = (
                    (rgb[:, :, 0] > 70)
                    & (rgb[:, :, 0] > 1.6 * rgb[:, :, 1])
                    & (rgb[:, :, 0] > 1.4 * rgb[:, :, 2])
                )
                prefix = f"episode-{ep}-tick-{tick}-wait-{wait}"
                Image.fromarray(rgb).save(output / f"{prefix}.png")
                Image.fromarray((cube * 255).astype(np.uint8)).save(
                    output / f"{prefix}-visible-cube.png"
                )
                np.savez_compressed(
                    output / f"{prefix}-rgbd.npz",
                    rgb=rgb,
                    depth=depth,
                    camera_position=pos,
                    camera_rotation=rot,
                    fovy=world.model.cam_fovy[camera],
                )
                orientation = world.data.body("red_block").xmat.reshape(3, 3)
                points, valid = point_cloud(rgb, depth, pos, rot, world.model.cam_fovy[camera])
                red &= valid
                yy, xx = np.indices(depth.shape)
                focal = depth.shape[0] / (2 * np.tan(np.deg2rad(world.model.cam_fovy[camera]) / 2))
                rays = (
                    np.stack(
                        (
                            (xx + 0.5 - depth.shape[1] / 2) / focal,
                            -(yy + 0.5 - depth.shape[0] / 2) / focal,
                            -np.ones_like(depth),
                        ),
                        axis=-1,
                    )
                    @ rot.reshape(3, 3).T
                )
                center = world.data.body("red_block").xpos.copy()
                box_depth = ray_box_depth(pos, rays, center, orientation, np.full(3, 0.025))
                interior = erode(cube)
                depth_error = np.abs(depth - box_depth)
                distance = surface_distance(points[red], center, orientation, np.full(3, 0.025))
                rr, gg, bb = rgb.astype(float).transpose(2, 0, 1)
                obj = points[red]
                wall_mask = valid & (gg > 70) & (gg > 1.3 * rr) & (bb > 1.2 * rr)
                wall_mask &= (
                    (points[:, :, 0] > obj[:, 0].max())
                    & (points[:, :, 1] >= obj[:, 1].min() - 0.005)
                    & (points[:, :, 1] <= obj[:, 1].max() + 0.005)
                    & (points[:, :, 2] > 0.04)
                )
                wall_ids = segmentation[:, :, 0][wall_mask]
                counts = {
                    mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or str(i): int(
                        np.sum(wall_ids == i)
                    )
                    for i in np.unique(wall_ids)
                    if i >= 0
                }
                measured_wall = float(points[wall_mask, 0].min()) if wall_mask.any() else None
                errors = {
                    "wait_ticks": wait,
                    "red_pixels": int(red.sum()),
                    "red_pixels_wrong_geom": int((red & ~cube).sum()),
                    "red_surface_distance": stats(distance),
                    "red_points_incompatible_with_3mm_per_axis": int(
                        np.sum(distance > np.sqrt(3) * 0.003)
                    ),
                    "depth_interior": stats(depth_error[interior]),
                    "depth_boundary": stats(depth_error[cube & ~interior]),
                    "segmentation_cube_rays_missing_box": int(
                        (cube & ~np.isfinite(box_depth)).sum()
                    ),
                    "wall_association_counts": counts,
                    "wall_x_error_m": None if measured_wall is None else measured_wall - 0.54,
                }
                report["cases"].append(
                    {
                        "episode": ep,
                        "error_audit": errors,
                        "takeover_tick": record["takeover"]["tick"],
                        "decision_tick": tick,
                        "recorded_success": record["success"],
                        "recorded_reason": record["reason"],
                        "detection": detection,
                        "error": error,
                        "diagnostics": diagnostic,
                        "visible_cube_pixels": int(cube.sum()),
                        "red_cube_pixels": int((red & cube).sum()),
                        "truth_diagnostic_only": {
                            "xyz": world.data.body("red_block").xpos.tolist(),
                            "nearest_face_tilt_degrees": float(
                                np.rad2deg(
                                    np.arccos(np.clip(np.max(np.abs(orientation[2, :])), 0, 1))
                                )
                            ),
                        },
                        "image": str(output / f"{prefix}.png"),
                        "rgbd_sha256": digest(output / f"{prefix}-rgbd.npz"),
                    }
                )
                print(json.dumps(report["cases"][-1]), flush=True)
            finally:
                if renderer is not None:
                    renderer.close()
                world.close()
        if payload_digest(root) != source:
            raise ValueError("dataset changed")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/evaluations/geometry-error-v1.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
