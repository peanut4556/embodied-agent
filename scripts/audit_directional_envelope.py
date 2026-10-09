"""Replay the fixed 54 RGB-D stress cases; truth is used only for scoring."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from embodied_agent.correction_data import payload_digest
from embodied_agent.directional_envelope import directional_envelope
from embodied_agent.memory_policy import digest
from embodied_agent.physics import PhysicsWorld
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
            corners = (
                np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
            )
            truths[ep] = (
                corners @ data.geom("red_block").xmat.reshape(3, 3).T + data.geom("red_block").xpos
            )
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
        reliable = points[filter_masks(image, depth)["combined"]]
        row = {
            k: case[k]
            for k in (
                "episode",
                "camera_shift_x_m",
                "occlusion",
                "depth_perturbation",
                "beyond_nominal_error_budget",
            )
        }
        row["reliable_points"] = len(reliable)
        if len(reliable) != case["estimate"]["reliable_points"]:
            raise ValueError("reliable points changed")
        if len(reliable) < 30:
            row.update(status="unknown", clearance_certified=False)
        else:
            result = directional_envelope(reliable)
            for key in ("aabb_min", "aabb_max"):
                np.testing.assert_allclose(
                    result["baseline"][key], case["estimate"]["bound"][key], atol=1e-12, rtol=0
                )
            truth = truths[ep]
            projected = truth @ np.array(result["directions"]).T
            contained = bool(
                np.all(truth >= result["aabb_min"])
                and np.all(truth <= result["aabb_max"])
                and np.all(projected >= result["projection_min"])
                and np.all(projected <= result["projection_max"])
            )
            old = case["estimate"]["bound"]
            gap = old["conditional_x_gap_lower_m"]
            if gap is not None:
                gap += old["aabb_max"][0] - result["aabb_max"][0]
            true_gap = float(0.54 - truth[:, 0].max())
            np.testing.assert_allclose(
                true_gap, case["scoring_only"]["true_x_gap_m"], atol=1e-10, rtol=0
            )
            row.update(
                status="conditional_bound",
                bound=result,
                scoring_only={
                    "contained": contained,
                    "true_x_gap_m": true_gap,
                    "false_positive_gap": bool(gap is not None and gap > 0 and true_gap <= 0),
                },
                conditional_x_gap_lower_m=gap,
            )
        cases.append(row)
    bounds = [c for c in cases if "bound" in c]
    if payload_digest(root) != source["source_sha256"]:
        raise ValueError("dataset changed during audit")
    # Matched witnesses, using only the observation-derived slabs for rejection.
    attribution_path = Path("docs/evaluations/overlap-attribution-v1.json")
    attribution = json.loads(attribution_path.read_text())
    if attribution["source_sha256"] != source["source_sha256"]:
        raise ValueError("witness dataset differs")
    selected = next(
        c["bound"]
        for c in cases
        if c["episode"] == 5
        and c["camera_shift_x_m"] == 0
        and c["occlusion"] == "none"
        and c["depth_perturbation"] == "none"
    )
    directions = np.array(selected["directions"])
    witness_rows = []
    world = PhysicsWorld()
    try:
        data = mujoco.MjData(world.model)
        data.qpos[:] = world.data.qpos
        for w in attribution["witnesses"]:
            if w["obstacle"] != "object_envelope":
                continue
            data.qpos[:5] = w["joints"]
            mujoco.mj_forward(world.model, data)
            geom = data.geom(w["geom"])
            center = directions @ geom.xpos
            radius = np.abs(directions @ geom.xmat.reshape(3, 3)) @ world.model.geom(w["geom"]).size
            separated = bool(
                np.any(center + radius < np.array(selected["projection_min"]) - 1e-9)
                or np.any(center - radius > np.array(selected["projection_max"]) + 1e-9)
            )
            witness_rows.append(
                {k: w[k] for k in ("path", "segment", "fraction", "geom", "category")}
                | {"separated_by_directional_slab": separated}
            )
    finally:
        world.close()
    report = {
        "status": "complete",
        "script_sha256": digest(__file__),
        "implementation_sha256": digest("src/embodied_agent/directional_envelope.py"),
        "source_report_sha256": digest(source_path),
        "source_sha256": source["source_sha256"],
        "model_sha256": manifest["model_sha256"],
        "summary": {
            "trials": len(cases),
            "bounds": len(bounds),
            "unknown": len(cases) - len(bounds),
            "containment_failures": sum(not c["scoring_only"]["contained"] for c in bounds),
            "false_positive_gaps": sum(c["scoring_only"]["false_positive_gap"] for c in bounds),
            "volume_reduction_min_max": [
                min(c["bound"]["aabb_volume_reduction_fraction"] for c in bounds),
                max(c["bound"]["aabb_volume_reduction_fraction"] for c in bounds),
            ],
        },
        "cases": cases,
        "witness_report_sha256": digest(attribution_path),
        "witness_screen": {
            "object_witnesses": len(witness_rows),
            "separated": sum(w["separated_by_directional_slab"] for w in witness_rows),
            "truth_overlap_wrongly_separated": sum(
                w["separated_by_directional_slab"] and w["category"] == "truth_overlap"
                for w in witness_rows
            ),
            "rows": witness_rows,
        },
        "group": "train",
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "clearance_certified": False,
        "limitations": [
            "diameter slabs do not establish true object orientation",
            "6mm bias exceeds nominal 3mm budget",
            "image masking only; fixed camera translations, not physical occluders",
            "single normal and single tilted training state; no continuous path rescreen",
        ],
    }
    Path("docs/evaluations/directional-envelope-v1.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            report["summary"] | {"witness_separated": report["witness_screen"]["separated"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
