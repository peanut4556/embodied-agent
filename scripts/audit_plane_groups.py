"""Fixed visible-face proposals and normal errors; no new orientation guarantee."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from embodied_agent.correction_data import payload_digest
from embodied_agent.memory_policy import digest
from embodied_agent.physics import PhysicsWorld
from embodied_agent.plane_groups import plane_groups
from embodied_agent.rgbd_filter import filter_masks
from embodied_agent.visible_geometry import point_cloud


def main():
    source_path = Path("docs/evaluations/envelope-stress-v1.json")
    source = json.loads(source_path.read_text())
    root = Path("outputs/datasets/corrections-contact-oriented-v1")
    manifest = json.loads((root / "recording.json").read_text())
    if (
        payload_digest(root) != source["source_sha256"]
        or manifest["status"] != "validated"
        or manifest["model_sha256"] != digest("src/embodied_agent/assets/tabletop.xml")
        or manifest["fps"] != 25
    ):
        raise ValueError("source changed")
    table = pq.read_table(
        sorted((root / "data").rglob("*.parquet")), columns=["replay.action", "observation.state"]
    )
    actions, states = table["replay.action"].to_pylist(), table["observation.state"].to_pylist()
    truths = {}
    for ep, wait in ((3, 0), (5, 100)):
        world = PhysicsWorld()
        try:
            record = manifest["episodes"][ep]
            data, model = world.data, world.model
            offset = sum(e["frames"] for e in manifest["episodes"][:ep])
            for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                getattr(data, key)[:] = record["initial_state"][key]
            data.time = record["initial_state"]["time"]
            tick = record["takeover"]["tick"] + 74
            for t in range(tick + 1):
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.qpos[:5], states[offset + t], atol=1e-6, rtol=0)
                if t < tick:
                    data.ctrl[:] = actions[offset + t]
                    for _ in range(20):
                        mujoco.mj_step(model, data)
            for _ in range(wait * 20):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            size = model.geom("red_block").size
            truths[ep] = {
                "center": data.geom("red_block").xpos.copy(),
                "rotation": data.geom("red_block").xmat.reshape(3, 3).copy(),
                "size": size.copy(),
            }
        finally:
            world.close()
    inputs = {i["path"]: i["sha256"] for i in source["inputs"]}
    cases = []
    for case in source["cases"]:
        ep, shift = case["episode"], case["camera_shift_x_m"]
        path = f"outputs/evaluations/envelope-stress-v1/episode-{ep}-camera-{shift:+.2f}.npz"
        if digest(path) != inputs[path]:
            raise ValueError("raw RGB-D changed")
        with np.load(path) as raw:
            image, depth = raw["rgb"].copy(), raw["depth"].copy()
            clean, _ = point_cloud(image, depth, raw["position"], raw["rotation"], raw["fovy"])
            r, g, b = image.astype(float).transpose(2, 0, 1)
            red = (r > 70) & (r > 1.6 * g) & (r > 1.4 * b)
            if case["occlusion"] == "half_red":
                _, xx = np.indices(depth.shape)
                image[red & (xx <= np.median(np.nonzero(red)[1]))] = 0
            elif case["occlusion"] == "all_red":
                image[red] = 0
            if case["depth_perturbation"] == "uniform_2mm":
                depth += np.random.default_rng(73).uniform(-0.002, 0.002, depth.shape)
            elif case["depth_perturbation"] == "bias_6mm":
                depth += 0.006
            points, _ = point_cloud(image, depth, raw["position"], raw["rotation"], raw["fovy"])
        mask = filter_masks(image, depth)["combined"]
        reliable = points[mask]
        if len(reliable) != case["estimate"]["reliable_points"]:
            raise ValueError("reliable points changed")
        result = plane_groups(reliable)
        truth = truths[ep]
        # Assign scoring labels using UNPERTURBED same-pixel points, so noise
        # does not silently relabel a point to a different cube face.
        local = (clean[mask] - truth["center"]) @ truth["rotation"]
        distances = np.abs(np.abs(local) - truth["size"])
        order = np.argsort(distances, axis=1)
        labels = np.full(len(local), -1, dtype=int)
        for i, (first, second, _) in enumerate(order):
            if (
                distances[i, first] <= 0.001
                and distances[i, second] - distances[i, first] > 0.001
                and np.all(np.abs(local[i]) <= truth["size"] + 0.001)
            ):
                labels[i] = 2 * first + int(local[i, first] >= 0)
        for group in result["groups"]:
            idx = group["indices"]
            assigned = labels[idx]
            counts = np.bincount(assigned[assigned >= 0], minlength=6)
            face = int(counts.argmax()) if counts.sum() else None
            normal = np.array(group["normal"])
            angle = (
                None
                if face is None
                else float(
                    np.rad2deg(
                        np.arccos(np.clip(abs(normal @ truth["rotation"][:, face // 2]), 0, 1))
                    )
                )
            )
            group["scoring_only"] = {
                "face_counts": counts.tolist(),
                "ambiguous_or_off_surface": int(np.sum(assigned < 0)),
                "dominant_face": face,
                "dominant_fraction_all_points": float(counts.max() / len(idx)),
                "normal_error_degrees": angle,
            }
            group["point_count"] = len(group.pop("indices"))
        result["unassigned_points"] = len(result.pop("unassigned_indices"))
        cases.append(
            {
                k: case[k]
                for k in (
                    "episode",
                    "camera_shift_x_m",
                    "occlusion",
                    "depth_perturbation",
                    "beyond_nominal_error_budget",
                )
            }
            | {"reliable_points": len(reliable), "result": result}
        )
    groups = [g for c in cases for g in c["result"]["groups"] if g["proposal_accepted"]]
    summaries = {}
    for noise in ("none", "uniform_2mm", "bias_6mm"):
        accepted = [
            g
            for c in cases
            if c["depth_perturbation"] == noise
            for g in c["result"]["groups"]
            if g["proposal_accepted"]
        ]
        angles = [
            g["scoring_only"]["normal_error_degrees"]
            for g in accepted
            if g["scoring_only"]["normal_error_degrees"] is not None
        ]
        summaries[noise] = {
            "accepted_groups": len(accepted),
            "normal_error_max_degrees": max(angles, default=None),
            "groups_below_95_percent_dominant": sum(
                g["scoring_only"]["dominant_fraction_all_points"] < 0.95 for g in accepted
            ),
        }
    if payload_digest(root) != source["source_sha256"]:
        raise ValueError("dataset changed")
    report = {
        "status": "complete",
        "script_sha256": digest(__file__),
        "implementation_sha256": digest("src/embodied_agent/plane_groups.py"),
        "source_report_sha256": digest(source_path),
        "source_sha256": source["source_sha256"],
        "model_sha256": manifest["model_sha256"],
        "group": "train",
        "parameters": {
            "seed": 73,
            "ransac_trials": 256,
            "max_groups": 3,
            "inlier_threshold_m": 0.0015,
            "minimum_points": 30,
            "minimum_tangent_span_m": 0.01,
            "maximum_residual_m": 0.002,
        },
        "summary": {
            "cases": len(cases),
            "accepted_groups": len(groups),
            "unknown_cases": sum(c["result"]["status"] == "unknown" for c in cases),
            "by_depth_perturbation": summaries,
        },
        "cases": cases,
        "angle_bound_certified": False,
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "clearance_certified": False,
        "limitations": [
            "empirical errors are not a certified angle bound",
            "nearest-face labels exclude ambiguous edges; labels are scoring-only",
            "sequential RANSAC may miss faces or split one face into multiple groups",
            "two training states, image masking and camera translation only",
            "6mm bias exceeds the original error budget; no envelope or policy update",
        ],
    }
    Path("docs/evaluations/plane-groups-v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
