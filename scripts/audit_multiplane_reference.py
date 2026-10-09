"""Synthetic layout experiment, not deployment or existing-image calibration."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.multiplane_reference import observability, residuals


def main():
    rays = np.array(
        [[x, y, -1.0] for x in np.linspace(-0.4, 0.4, 25) for y in np.linspace(-0.3, 0.3, 25)]
    )
    root2 = np.sqrt(2)
    table = (np.array([0.0, 0.0, 1.0]), 0.0)
    tilted_x = (np.array([1.0, 0.0, 1.0]) / root2, 0.4 / root2)
    tilted_y = (np.array([0.0, 1.0, 1.0]) / root2, 0.0)
    layouts = {
        "horizontal": [table],
        "horizontal_plus_x": [table, tilted_x],
        "horizontal_plus_x_y": [table, tilted_x, tilted_y],
    }
    plan = [("control", np.zeros(7))]
    for i, name in enumerate(
        ("camera_x", "camera_y", "camera_z", "roll", "pitch", "yaw", "depth_bias")
    ):
        values = (-np.deg2rad(1), np.deg2rad(1)) if 3 <= i <= 5 else (-0.006, 0.006)
        for value in values:
            p = np.zeros(7)
            p[i] = value
            plan.append((f"{name}:{value:+.6f}", p))
    p = np.zeros(7)
    p[2] = p[6] = 0.006
    plan.append(("depth_height_cancel", p))
    reports = {}
    for name, planes in layouts.items():
        cases = []
        for label, parameters in plan:
            residual = residuals(rays, planes, parameters)
            p95 = np.quantile(np.abs(residual), 0.95, axis=1)
            cases.append(
                {
                    "name": label,
                    "parameters": parameters.tolist(),
                    "per_plane_p95_m": p95.tolist(),
                    "status": "alarm" if np.any(p95 > 0.003) else "no_alarm",
                }
            )
        reports[name] = {
            "planes": [{"normal": n.tolist(), "offset_m": d} for n, d in planes],
            "observability": observability(rays, planes),
            "cases": cases,
        }
    report = {
        "status": "complete",
        "experiment": "synthetic_known_plane_layouts",
        "script_sha256": digest(__file__),
        "implementation_sha256": digest("src/embodied_agent/multiplane_reference.py"),
        "parameters": {
            "camera_position": [0.4, 0, 1.2],
            "ray_grid": [25, 25],
            "ray_x_range": [-0.4, 0.4],
            "ray_y_range": [-0.3, 0.3],
            "parameter_order": [
                "tx_m",
                "ty_m",
                "tz_m",
                "rx_rad",
                "ry_rad",
                "rz_rad",
                "depth_bias_m",
            ],
            "alarm_threshold_m": 0.003,
            "finite_difference_step": 1e-6,
            "relative_rank_threshold": 1e-7,
        },
        "layouts": reports,
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "clearance_certified": False,
        "limitations": [
            "ideal exact infinite planes, known correspondences and no noise",
            "each plane is a separate calibration exposure; not intersecting visible surfaces in one image",
            "rank is local identifiability in this seven-parameter constant-bias model only",
            "full rank does not guarantee threshold detection or real sensor calibration",
            "spatially local object errors and arbitrary depth fields are not modeled",
            "no existing scene geometry, images or policy changed",
        ],
    }
    Path("docs/evaluations/multiplane-reference-v1.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for name, r in reports.items():
        print(name, "rank", r["observability"]["rank"], "cancel", r["cases"][-1])


if __name__ == "__main__":
    main()
