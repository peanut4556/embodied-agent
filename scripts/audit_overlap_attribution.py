"""Matched static witness ablation; truth is used only for offline scoring."""

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from embodied_agent.box_overlap import aabb_as_obb, obb_overlap
from embodied_agent.correction_data import payload_digest
from embodied_agent.gripper_sweep import box_aabb, overlaps
from embodied_agent.memory_policy import digest
from embodied_agent.physics import PhysicsWorld


def geom_box(model, data, name):
    return (
        data.geom(name).xpos.copy(),
        data.geom(name).xmat.reshape(3, 3).copy(),
        model.geom(name).size.copy(),
    )


def main():
    paths = ["gripper-sweep-refined-v1.json", "support-reach-v1.json", "envelope-stress-v1.json"]
    refined, support, stress = [json.loads(Path("docs/evaluations", p).read_text()) for p in paths]
    for p, sha in refined["source_reports"].items():
        if digest(Path("docs/evaluations", p)) != sha:
            raise ValueError("source report changed")
    root = Path("outputs/datasets/corrections-contact-oriented-v1")
    manifest = json.loads((root / "recording.json").read_text())
    source = payload_digest(root)
    if (
        source != support["source_sha256"]
        or source != stress["source_sha256"]
        or manifest["status"] != "validated"
        or manifest["fps"] != 25
        or json.loads((root / "selection.json").read_text())["payload_sha256"] != source
        or digest("src/embodied_agent/assets/tabletop.xml") != support["model_sha256"]
    ):
        raise ValueError("source data or model changed")
    table = pq.read_table(
        sorted((root / "data").rglob("*.parquet")), columns=["replay.action", "observation.state"]
    )
    actions = table["replay.action"].to_pylist()
    states = table["observation.state"].to_pylist()
    record = manifest["episodes"][5]
    offset = sum(e["frames"] for e in manifest["episodes"][:5])
    selected = [
        c
        for c in stress["cases"]
        if c["episode"] == 5
        and c["camera_shift_x_m"] == 0
        and c["occlusion"] == "none"
        and c["depth_perturbation"] == "none"
    ]
    if len(selected) != 1:
        raise ValueError("ambiguous envelope")
    bound = selected[0]["estimate"]["bound"]
    envelope = aabb_as_obb((bound["aabb_min"], bound["aabb_max"]))
    rows = []
    world = PhysicsWorld()
    try:
        model, live = world.model, world.data
        for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
            getattr(live, key)[:] = record["initial_state"][key]
        live.time = record["initial_state"]["time"]
        for tick in range(159):
            mujoco.mj_forward(model, live)
            np.testing.assert_allclose(live.qpos[:5], states[offset + tick], atol=1e-6, rtol=0)
            if tick < 158:
                live.ctrl[:] = actions[offset + tick]
                for _ in range(20):
                    mujoco.mj_step(model, live)
        for _ in range(2000):
            mujoco.mj_step(model, live)
        mujoco.mj_forward(model, live)
        truth = geom_box(model, live, "red_block")
        truth_bounds = box_aabb(*truth)
        np.testing.assert_allclose(truth[0], support["support"][-1]["xyz"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            truth_bounds,
            [support["geometry"]["cube_aabb_min"], support["geometry"]["cube_aabb_max"]],
            atol=1e-10,
            rtol=0,
        )
        if not (
            np.all(truth_bounds[0] >= bound["aabb_min"])
            and np.all(truth_bounds[1] <= bound["aabb_max"])
        ):
            raise ValueError("truth not contained by envelope")
        original = {k: getattr(live, k).copy() for k in ("qpos", "qvel", "ctrl")}
        data = mujoco.MjData(model)
        data.qpos[:] = live.qpos
        center = support["support"][-1]["xyz"][0]
        for path in refined["paths"]:
            name = path["name"]
            dx = 0 if name == "high_only" else float(name.removeprefix("descend-"))
            points = [(0.22, 0.30), (center + dx, 0.30)]
            if name != "high_only":
                points.append((center + dx, 0.03))
            qs = []
            for a, b in pairwise(points):
                count = int(np.ceil(np.linalg.norm(np.array(b) - a) / 0.01))
                for t in np.linspace(0, 1, count + 1)[:-1]:
                    qs.append(np.r_[world.ik(*(np.array(a) + t * (np.array(b) - a))), 0.04, 0.04])
            qs.append(np.r_[world.ik(*points[-1]), 0.04, 0.04])
            if len(qs) != path["waypoints"]:
                raise ValueError("path changed")
            for interval in path["refinement"]["intervals"]:
                if interval["status"] != "static_envelope_overlap":
                    continue
                # One fixed earliest recorded witness per coarse combination, no search.
                t = min(interval["static_witness_fractions"])
                i, geom, obstacle = interval["segment"], interval["geom"], interval["obstacle"]
                q = qs[i] + t * (qs[i + 1] - qs[i])
                if np.any(q < model.actuator_ctrlrange[:, 0]) or np.any(
                    q > model.actuator_ctrlrange[:, 1]
                ):
                    raise ValueError("joint limit")
                data.qpos[:5] = q
                mujoco.mj_forward(model, data)
                gripper = geom_box(model, data, geom)
                actual_name = "red_block" if obstacle == "object_envelope" else obstacle
                actual = geom_box(model, data, actual_name)
                supplied = envelope if obstacle == "object_envelope" else actual
                ga = aabb_as_obb(box_aabb(*gripper))
                if not overlaps(box_aabb(*gripper), box_aabb(*supplied)):
                    raise ValueError("witness no longer overlaps")
                exact_supplied = obb_overlap(gripper, supplied)
                aabb_truth = obb_overlap(ga, actual)
                exact_truth = obb_overlap(gripper, actual)
                distance = mujoco.mj_geomDistance(
                    model, data, model.geom(geom).id, model.geom(actual_name).id, 1.0, None
                )
                if abs(distance) > 1e-7 and exact_truth != (distance < 0):
                    raise ValueError("SAT disagrees with MuJoCo signed distance")
                if exact_truth and not exact_supplied:
                    raise ValueError("nested volume inconsistency")
                category = (
                    "gripper_aabb_only"
                    if not exact_supplied
                    else "object_envelope_only"
                    if not exact_truth
                    else "truth_overlap"
                )
                rows.append(
                    {
                        "path": name,
                        "segment": i,
                        "fraction": t,
                        "geom": geom,
                        "obstacle": obstacle,
                        "joints": q.tolist(),
                        "aabb_supplied": True,
                        "obb_supplied": exact_supplied,
                        "aabb_truth": aabb_truth,
                        "obb_truth": exact_truth,
                        "truth_signed_distance_m": float(distance),
                        "category": category,
                    }
                )
        for k, v in original.items():
            np.testing.assert_array_equal(getattr(live, k), v)
    finally:
        world.close()
    if payload_digest(root) != source:
        raise ValueError("dataset changed")
    report = {
        "status": "complete",
        "group": "train",
        "source_sha256": source,
        "model_sha256": support["model_sha256"],
        "script_sha256": digest(__file__),
        "geometry_sha256": digest("src/embodied_agent/box_overlap.py"),
        "source_reports": {p: digest(Path("docs/evaluations", p)) for p in paths},
        "scoring_only_truth": {
            "center": truth[0].tolist(),
            "rotation": truth[1].tolist(),
            "half_size": truth[2].tolist(),
        },
        "summary": dict(Counter(r["category"] for r in rows)),
        "paths": {
            p["name"]: dict(Counter(r["category"] for r in rows if r["path"] == p["name"]))
            for p in refined["paths"]
        },
        "witnesses": rows,
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "clearance_certified": False,
        "limitations": [
            "static witnesses only, not a rescreened continuous path",
            "ordered attribution: gripper AABB first, then object envelope",
            "truth used only for scoring, not control or permission to move",
            "single training fixture; no complete arm or unseen obstacles",
        ],
    }
    Path("docs/evaluations/overlap-attribution-v1.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report["paths"], indent=2))


if __name__ == "__main__":
    main()
