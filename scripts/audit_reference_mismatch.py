"""Fixed missing-depth, foreground, calibration and cancellation controls."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.reference_plane import check_reference, reference_geometry


def main():
    source_path = Path("docs/evaluations/envelope-stress-v1.json")
    previous_path = Path("docs/evaluations/reference-plane-v1.json")
    source, previous = [json.loads(p.read_text()) for p in (source_path, previous_path)]
    if digest(source_path) != previous["source_report_sha256"]:
        raise ValueError("source changed")
    plan = [("control", 0.0)]
    plan += [
        (kind, fraction) for kind in ("missing", "foreground") for fraction in (0.1, 0.25, 1.0)
    ]
    plan += [("camera_z", value) for value in (-0.006, -0.002, 0.002, 0.006)]
    plan += [("camera_pitch_degrees", value) for value in (-1.0, -0.25, 0.25, 1.0)]
    plan += [(axis, value) for axis in ("camera_x", "camera_y") for value in (-0.02, 0.02)]
    plan += [("depth_bias", value) for value in (-0.006, 0.006)]
    plan += [("depth_and_height_cancel", 0.006)]
    cases = []
    for item in source["inputs"]:
        path = item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError("raw RGB-D changed")
        with np.load(path) as raw:
            depth = raw["depth"].copy()
            position, rotation = raw["position"].copy(), raw["rotation"].reshape(3, 3).copy()
            fovy = float(raw["fovy"])
        _, roi = reference_geometry(depth.shape, position, rotation, fovy)
        ids = np.flatnonzero(roi)  # Fixed row-major prefix, no observed-depth selection.
        for kind, value in plan:
            changed = depth.copy()
            pos, rot = position.copy(), rotation.copy()
            changed_pixels = 0
            if kind in ("missing", "foreground"):
                chosen = ids[: int(np.ceil(len(ids) * value))]
                changed_pixels = len(chosen)
                if kind == "missing":
                    changed.flat[chosen] = np.nan
                else:
                    changed.flat[chosen] -= 0.020
            elif kind == "camera_pitch_degrees":
                angle = np.deg2rad(value)
                c, s = np.cos(angle), np.sin(angle)
                rot = rotation @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            elif kind in ("camera_x", "camera_y", "camera_z"):
                pos[{"camera_x": 0, "camera_y": 1, "camera_z": 2}[kind]] += value
            elif kind == "depth_bias":
                changed += value
            elif kind == "depth_and_height_cancel":
                changed += value
                pos[2] += value
            result = check_reference(changed, pos, rot, fovy)
            if kind == "control":
                old = next(
                    c["reference"]
                    for c in previous["cases"]
                    if c["scope"] == "none"
                    and path.endswith(
                        f"episode-{c['episode']}-camera-{c['camera_shift_x_m']:+.2f}.npz"
                    )
                )
                if result != old:
                    raise ValueError("refactor changed baseline behavior")
            cases.append(
                {
                    "input": path,
                    "perturbation": kind,
                    "value": value,
                    "modified_reference_pixels": changed_pixels,
                    "reference": result,
                }
            )
    summary = {}
    for kind, value in plan:
        rows = [c["reference"] for c in cases if c["perturbation"] == kind and c["value"] == value]
        residuals = [r["absolute_residual_p95_m"] for r in rows if "absolute_residual_p95_m" in r]
        summary[f"{kind}:{value:g}"] = {
            "cases": len(rows),
            "statuses": {
                s: sum(r["status"] == s for r in rows) for s in ("alarm", "no_alarm", "unknown")
            },
            "p95_range_m": [min(residuals), max(residuals)] if residuals else None,
        }
    report = {
        "status": "complete",
        "script_sha256": digest(__file__),
        "implementation_sha256": digest("src/embodied_agent/reference_plane.py"),
        "source_report_sha256": digest(source_path),
        "baseline_sha256": digest(previous_path),
        "parameters": {
            "foreground_depth_offset_m": -0.020,
            "residual_p95_threshold_m": 0.003,
            "minimum_valid_fraction": 0.8,
            "plan": plan,
        },
        "summary": summary,
        "cases": cases,
        "group": "train",
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "clearance_certified": False,
        "limitations": [
            "missing depth is not equivalent to a foreground return",
            "foreground is a synthetic depth overlay, not a rendered occluder",
            "extrinsics are metadata errors, not a physically moved camera",
            "alarm has no unique causal interpretation; no automatic correction",
            "same known planar fixture and six training images; local object errors remain unobserved",
        ],
    }
    Path("docs/evaluations/reference-mismatch-v1.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
