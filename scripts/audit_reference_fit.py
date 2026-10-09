"""Finite angular patches, bounded noise, and wrong plane locations; offline only."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.reference_fit import fit_reference, predict


def main():
    root2 = np.sqrt(2)
    planes = [
        (np.array([0.0, 0.0, 1.0]), 0.0),
        (np.array([1.0, 0.0, 1.0]) / root2, 0.4 / root2),
        (np.array([0.0, 1.0, 1.0]) / root2, 0.0),
    ]
    truth = np.r_[[0.002, -0.003, 0.004], np.deg2rad([0.2, -0.15, 0.1]), 0.002]
    shift = np.array([0.001, -0.001, 0.002])
    offsets = {
        "none": np.zeros(3),
        "independent_mm": np.array([0.001, -0.001, 0.001]),
        "common_normal_mm": np.full(3, 0.001),
        "rigid_layout_shift": np.array([n @ shift for n, d in planes]),
    }
    cases = []
    for patch, extent in (
        ("wide", (0.25, 0.20)),
        ("medium", (0.08, 0.06)),
        ("narrow", (0.025, 0.02)),
    ):
        rays = np.array(
            [
                [x, y, -1.0]
                for x in np.linspace(-extent[0], extent[0], 21)
                for y in np.linspace(-extent[1], extent[1], 21)
            ]
        )
        for noise in (0.0, 0.001, 0.003):
            for seed in (73,) if noise == 0 else (73, 91, 117):
                perturbation = np.random.default_rng(seed).uniform(-noise, noise, (3, len(rays)))
                for label, offset in offsets.items():
                    actual_planes = [
                        (n, d + delta) for (n, d), delta in zip(planes, offset, strict=True)
                    ]
                    observed = predict(rays, actual_planes, truth) + perturbation
                    fit = fit_reference(rays, planes, observed)
                    error = np.array(fit["parameters"]) - truth
                    cases.append(
                        {
                            "patch": patch,
                            "ray_half_widths": list(extent),
                            "noise_bound_m": noise,
                            "seed": seed,
                            "plane_error": label,
                            "plane_offset_errors_m": offset.tolist(),
                            "fit": fit,
                            "scoring_only": {
                                "translation_error_norm_m": float(np.linalg.norm(error[:3])),
                                "rotation_vector_error_norm_degrees": float(
                                    np.rad2deg(np.linalg.norm(error[3:6]))
                                ),
                                "depth_bias_error_m": float(error[6]),
                            },
                        }
                    )
    summary = {}
    for patch in ("wide", "medium", "narrow"):
        rows = [c for c in cases if c["patch"] == patch]
        summary[patch] = {
            "cases": len(rows),
            "statuses": {
                s: sum(c["fit"]["status"] == s for c in rows)
                for s in sorted({c["fit"]["status"] for c in rows})
            },
            "condition_number_range": [
                min(c["fit"]["scaled_condition_number"] for c in rows),
                max(c["fit"]["scaled_condition_number"] for c in rows),
            ],
            "max_translation_error_m": max(
                c["scoring_only"]["translation_error_norm_m"] for c in rows
            ),
            "max_rotation_vector_error_degrees": max(
                c["scoring_only"]["rotation_vector_error_norm_degrees"] for c in rows
            ),
            "max_absolute_depth_bias_error_m": max(
                abs(c["scoring_only"]["depth_bias_error_m"]) for c in rows
            ),
        }
    report = {
        "status": "complete",
        "experiment": "synthetic_finite_patch_calibration",
        "script_sha256": digest(__file__),
        "implementation_sha256": digest("src/embodied_agent/reference_fit.py"),
        "geometry_sha256": digest("src/embodied_agent/multiplane_reference.py"),
        "true_parameters": truth.tolist(),
        "summary": summary,
        "cases": cases,
        "parameters": {
            "samples_per_plane": 441,
            "max_iterations": 20,
            "max_backtracks": 12,
            "translation_and_bias_limit_m": 0.05,
            "rotation_component_limit_degrees": 5,
            "relative_rank_threshold": 1e-7,
            "fixed_seeds": [73, 91, 117],
        },
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "calibration_certified": False,
        "limitations": [
            "finite ray patches with exact correspondence; no physical occlusion or registration uncertainty",
            "same 441 samples per plane across patch widths, isolates spatial extent rather than pixel count",
            "plane normal errors are not included; offsets only",
            "error norms are scoring-only; low residual is not proof of correct extrinsics",
            "all fits including stalled/limited results retained; no online correction",
        ],
    }
    Path("docs/evaluations/reference-fit-v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
