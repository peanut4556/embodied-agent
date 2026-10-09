"""Object-local versus frame-wide slopes, with a fixed independent reference ROI."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.depth_ramp import depth_ramp
from embodied_agent.memory_policy import digest
from embodied_agent.reference_plane import check_reference


def main():
    baseline_path = Path("docs/evaluations/depth-ramp-v1.json")
    baseline = json.loads(baseline_path.read_text())
    source_path = Path("docs/evaluations/envelope-stress-v1.json")
    source = json.loads(source_path.read_text())
    if (
        digest(source_path) != baseline["source_report_sha256"]
        or digest("src/embodied_agent/depth_ramp.py") != baseline["perturbation_sha256"]
        or digest("src/embodied_agent/assets/tabletop.xml") != baseline["model_sha256"]
    ):
        raise ValueError("source changed")
    cases = []
    for ep in (3, 5):
        for shift in (-0.08, 0.0, 0.08):
            path = f"outputs/evaluations/envelope-stress-v1/episode-{ep}-camera-{shift:+.2f}.npz"
            expected_hash = next(i["sha256"] for i in source["inputs"] if i["path"] == path)
            if digest(path) != expected_hash:
                raise ValueError("RGB-D changed")
            with np.load(path) as raw:
                image, depth = raw["rgb"], raw["depth"]
                r, g, b = image.astype(float).transpose(2, 0, 1)
                red = (r > 70) & (r > 1.6 * g) & (r > 1.4 * b)
                _, x = np.indices(depth.shape)
                xs = x[red]
                profile = np.clip(
                    (x - (xs.min() + xs.max()) / 2) / ((xs.max() - xs.min()) / 2), -1, 1
                )
                plan = [("none", 0.0)] + [
                    (scope, a)
                    for scope in ("object_only", "whole_frame")
                    for a in (-0.006, -0.002, 0.002, 0.006)
                ]
                for scope, amplitude in plan:
                    object_changed, _ = depth_ramp(depth, red, amplitude, "shared")
                    changed = (
                        depth + amplitude * profile if scope == "whole_frame" else object_changed
                    )
                    # Identical pixel values on the original object; truth scores
                    # from the previous shared-slope groups are context, not a
                    # claim that frame-wide filtering/proposals were rerun.
                    np.testing.assert_allclose(
                        changed[red], object_changed[red], atol=1e-12, rtol=0
                    )
                    reference = check_reference(
                        changed, raw["position"], raw["rotation"], float(raw["fovy"])
                    )
                    old = next(
                        c
                        for c in baseline["cases"]
                        if c["episode"] == ep
                        and c["camera_shift_x_m"] == shift
                        and c["occlusion"] == "none"
                        and c["amplitude_m"] == amplitude
                        and c["depth_perturbation"] == ("none" if scope == "none" else "shared")
                    )
                    high = sum(
                        g["holdout"]["status"] == "consistent_candidate"
                        and g["scoring_only"]["normal_error_degrees"] > 5
                        for g in old["result"]["groups"]
                    )
                    cases.append(
                        {
                            "episode": ep,
                            "camera_shift_x_m": shift,
                            "scope": scope,
                            "amplitude_m": amplitude,
                            "reference": reference,
                            "prior_object_only_high_error_groups": high,
                        }
                    )
    summary = {}
    for scope in ("none", "object_only", "whole_frame"):
        for magnitude in (0.0,) if scope == "none" else (0.002, 0.006):
            rows = [c for c in cases if c["scope"] == scope and abs(c["amplitude_m"]) == magnitude]
            summary[f"{scope}-{magnitude:.3f}"] = {
                "cases": len(rows),
                "statuses": {
                    s: sum(c["reference"]["status"] == s for c in rows)
                    for s in ("alarm", "no_alarm", "unknown")
                },
                "p95_absolute_residual_range_m": [
                    min(c["reference"]["absolute_residual_p95_m"] for c in rows),
                    max(c["reference"]["absolute_residual_p95_m"] for c in rows),
                ],
                "prior_high_error_groups_in_no_alarm_cases": sum(
                    c["prior_object_only_high_error_groups"]
                    for c in rows
                    if c["reference"]["status"] == "no_alarm"
                ),
            }
    report = {
        "status": "complete",
        "script_sha256": digest(__file__),
        "implementation_sha256": digest("src/embodied_agent/reference_plane.py"),
        "source_report_sha256": digest(source_path),
        "baseline_sha256": digest(baseline_path),
        "model_sha256": baseline["model_sha256"],
        "summary": summary,
        "cases": cases,
        "parameters": {
            "known_plane_world_z_m": 0,
            "reference_x_m": [0.20, 0.65],
            "reference_y_m": [0.20, 0.30],
            "residual_p95_threshold_m": 0.003,
            "minimum_pixels": 100,
            "minimum_valid_fraction": 0.8,
        },
        "group": "train",
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "clearance_certified": False,
        "limitations": [
            "known empty ROI and camera calibration assumed; not observation-only plane discovery",
            "reference-only detection, no depth correction or policy change",
            "object-local slopes can evade an unaffected reference",
            "frame-wide slopes saturate outside original object X range",
            "object scores are historical context; no frame-wide policy/holdout rerun",
        ],
    }
    Path("docs/evaluations/reference-plane-v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
