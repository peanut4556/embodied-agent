"""Matched candidate-normal sensitivity; original truth scores are read-only."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.plane_groups import plane_groups
from embodied_agent.plane_stability import normal_stability
from embodied_agent.rgbd_filter import filter_masks
from embodied_agent.visible_geometry import point_cloud


def main():
    baseline_path = Path("docs/evaluations/plane-groups-v1.json")
    baseline = json.loads(baseline_path.read_text())
    source_path = Path("docs/evaluations/envelope-stress-v1.json")
    source = json.loads(source_path.read_text())
    if (
        digest(source_path) != baseline["source_report_sha256"]
        or digest("src/embodied_agent/plane_groups.py") != baseline["implementation_sha256"]
    ):
        raise ValueError("baseline source changed")
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
        mask = filter_masks(image, depth)["combined"]
        reliable = points[mask]
        if len(reliable) != case["estimate"]["reliable_points"]:
            raise ValueError("reliable points changed")
        result = plane_groups(reliable)
        old = baseline["cases"][len(cases)]
        if any(
            old[k] != case[k]
            for k in ("episode", "camera_shift_x_m", "occlusion", "depth_perturbation")
        ):
            raise ValueError("case order changed")
        if len(result["groups"]) != len(old["result"]["groups"]):
            raise ValueError("group count changed")
        rows = []
        for group, previous in zip(result["groups"], old["result"]["groups"], strict=True):
            comparable = {k: v for k, v in group.items() if k != "indices"}
            expected = {
                k: v for k, v in previous.items() if k not in ("point_count", "scoring_only")
            }
            if comparable != expected or len(group["indices"]) != previous["point_count"]:
                raise ValueError("candidate differs from baseline")
            if group["proposal_accepted"]:
                rows.append(
                    {
                        "point_count": len(group["indices"]),
                        "scoring_only": previous["scoring_only"],
                        "stability": normal_stability(reliable, group["indices"]),
                    }
                )
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
            | {"groups": rows}
        )

    summary = {}
    for noise in ("none", "uniform_2mm", "bias_6mm"):
        groups = [g for c in cases if c["depth_perturbation"] == noise for g in c["groups"]]
        scores = [g for g in groups if g["scoring_only"]["normal_error_degrees"] is not None]
        summary[noise] = {
            "groups": len(groups),
            "statuses": {
                s: sum(g["stability"]["status"] == s for g in groups)
                for s in ("stable_candidate", "unstable", "unknown")
            },
            "low_error_at_most_3_degrees": sum(
                g["scoring_only"]["normal_error_degrees"] <= 3 for g in scores
            ),
            "low_error_not_retained": sum(
                g["scoring_only"]["normal_error_degrees"] <= 3
                and g["stability"]["status"] != "stable_candidate"
                for g in scores
            ),
            "high_error_above_5_degrees": sum(
                g["scoring_only"]["normal_error_degrees"] > 5 for g in scores
            ),
            "high_error_retained": sum(
                g["scoring_only"]["normal_error_degrees"] > 5
                and g["stability"]["status"] == "stable_candidate"
                for g in scores
            ),
        }
    report = {
        "status": "complete",
        "script_sha256": digest(__file__),
        "implementation_sha256": digest("src/embodied_agent/plane_stability.py"),
        "baseline_sha256": digest(baseline_path),
        "source_report_sha256": digest(source_path),
        "parameters": {
            "spatial_halves": 4,
            "all_point_thresholds_m": [0.001, 0.002],
            "minimum_points": 20,
            "minimum_span_m": 0.005,
            "instability_threshold_degrees": 5,
            "scoring_low_error_degrees": 3,
            "scoring_high_error_degrees": 5,
        },
        "summary": summary,
        "cases": cases,
        "group": "train",
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "angle_bound_certified": False,
        "clearance_certified": False,
        "limitations": [
            "within-observation sensitivity, not independent acquisition",
            "spatial halves and reselected inliers overlap or share selection bias",
            "a stable biased plane can pass; missing evidence stays unknown",
            "two training states; 6mm bias exceeds original error budget",
            "no envelope or control policy changed",
        ],
    }
    Path("docs/evaluations/plane-stability-v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
