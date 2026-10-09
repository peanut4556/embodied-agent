"""Matched depth-slope cases, scored without changing candidate or holdout thresholds."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.cube_consistency import cube_consistency
from embodied_agent.depth_ramp import depth_ramp
from embodied_agent.memory_policy import digest
from embodied_agent.plane_holdout import heldout_groups
from embodied_agent.rgbd_filter import filter_masks
from embodied_agent.visible_geometry import point_cloud


def main():
    baseline_path = Path("docs/evaluations/depth-ramp-v1.json")
    baseline = json.loads(baseline_path.read_text())
    source_path = Path("docs/evaluations/envelope-stress-v1.json")
    source = json.loads(source_path.read_text())
    if (
        digest(source_path) != baseline["source_report_sha256"]
        or digest("src/embodied_agent/plane_holdout.py") != baseline["implementation_sha256"]
        or digest("src/embodied_agent/plane_groups.py")
        != baseline["proposal_implementation_sha256"]
        or digest("src/embodied_agent/depth_ramp.py") != baseline["perturbation_sha256"]
    ):
        raise ValueError("source implementation changed")
    inputs = {i["path"]: i["sha256"] for i in source["inputs"]}
    cases = []
    for case in baseline["cases"]:
        ep, shift = case["episode"], case["camera_shift_x_m"]
        path = f"outputs/evaluations/envelope-stress-v1/episode-{ep}-camera-{shift:+.2f}.npz"
        if digest(path) != inputs[path]:
            raise ValueError("RGB-D changed")
        with np.load(path) as raw:
            image, depth = raw["rgb"].copy(), raw["depth"].copy()
            r, g, b = image.astype(float).transpose(2, 0, 1)
            red = (r > 70) & (r > 1.6 * g) & (r > 1.4 * b)
            if case["occlusion"] == "half_red":
                _, xx = np.indices(depth.shape)
                image[red & (xx <= np.median(np.nonzero(red)[1]))] = 0
            elif case["occlusion"] == "all_red":
                image[red] = 0
            depth, _ = depth_ramp(depth, red, case["amplitude_m"], case["depth_perturbation"])
            points, _ = point_cloud(image, depth, raw["position"], raw["rotation"], raw["fovy"])
        mask = filter_masks(image, depth)["combined"]
        result = heldout_groups(points, mask)
        previous = case["result"]["groups"]
        if len(result["groups"]) != len(previous) or int(mask.sum()) != case["reliable_points"]:
            raise ValueError("case changed")
        for group, old in zip(result["groups"], previous, strict=True):
            if {k: v for k, v in group.items() if k != "indices"} != {
                k: v for k, v in old.items() if k not in ("scoring_only", "point_count")
            } or len(group["indices"]) != old["point_count"]:
                raise ValueError("candidate changed")
        geometry = cube_consistency(points[mask], result["groups"])
        alarm = any(
            geometry.get(k, False)
            for k in ("size_contradiction", "face_size_alarm", "relation_alarm")
        )
        cases.append(
            {
                k: case[k]
                for k in (
                    "episode",
                    "camera_shift_x_m",
                    "occlusion",
                    "depth_perturbation",
                    "amplitude_m",
                    "beyond_nominal_error_budget",
                )
            }
            | {"geometry": geometry, "diagnostic_alarm": alarm, "original_groups": previous}
        )
    summary = {}
    for mode in ("none", "shared", "candidate", "validation"):
        for magnitude in (0.0,) if mode == "none" else (0.002, 0.006):
            selected = [
                c
                for c in cases
                if c["depth_perturbation"] == mode and abs(c["amplitude_m"]) == magnitude
            ]
            high = [
                (c, g)
                for c in selected
                for g in c["original_groups"]
                if g["holdout"]["status"] == "consistent_candidate"
                and g["scoring_only"]["normal_error_degrees"] is not None
                and g["scoring_only"]["normal_error_degrees"] > 5
            ]
            low = [
                (c, g)
                for c in selected
                for g in c["original_groups"]
                if g["holdout"]["status"] == "consistent_candidate"
                and g["scoring_only"]["normal_error_degrees"] is not None
                and g["scoring_only"]["normal_error_degrees"] <= 3
            ]
            summary[f"{mode}-{magnitude:.3f}"] = {
                "cases": len(selected),
                "size_contradiction_cases": sum(
                    c["geometry"].get("size_contradiction", False) for c in selected
                ),
                "face_size_alarm_cases": sum(
                    c["geometry"].get("face_size_alarm", False) for c in selected
                ),
                "relation_alarm_cases": sum(
                    c["geometry"].get("relation_alarm", False) for c in selected
                ),
                "multiple_face_cases": sum(
                    c["geometry"].get("multiple_face_evidence") == "available" for c in selected
                ),
                "retained_high_error_groups": len(high),
                "high_error_groups_in_alarm_cases": sum(c["diagnostic_alarm"] for c, g in high),
                "retained_low_error_groups": len(low),
                "low_error_groups_in_alarm_cases": sum(c["diagnostic_alarm"] for c, g in low),
            }
    report = {
        "status": "complete",
        "script_sha256": digest(__file__),
        "implementation_sha256": digest("src/embodied_agent/cube_consistency.py"),
        "baseline_sha256": digest(baseline_path),
        "summary": summary,
        "cases": cases,
        "parameters": {
            "side_upper_m": 0.05,
            "coordinate_error_m": 0.003,
            "relation_alarm_degrees": 10,
        },
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "clearance_certified": False,
        "group": "train",
        "limitations": [
            "size checks are necessary, not sufficient for a cube fit",
            "face-size checks assume same-face membership; angle alarms are heuristic",
            "parallel normals may be repeated proposals of the same face",
            "single-face cases cannot test multi-face geometry",
            "6mm slopes exceed the point error budget; no strategy changed",
        ],
    }
    Path("docs/evaluations/cube-consistency-v1.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
