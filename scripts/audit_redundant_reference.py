"""Fourth-plane holdout and joint fits with fixed noise and location errors."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.redundant_reference import check_redundant_reference
from embodied_agent.reference_fit import fit_reference, predict


def main():
    baseline_path = Path("docs/evaluations/reference-fit-v1.json")
    baseline = json.loads(baseline_path.read_text())
    if (
        digest("src/embodied_agent/reference_fit.py") != baseline["implementation_sha256"]
        or digest("src/embodied_agent/multiplane_reference.py") != baseline["geometry_sha256"]
    ):
        raise ValueError("baseline implementation changed")
    planes = [
        (np.array([0.0, 0.0, 1.0]), 0.0),
        (np.array([1.0, 0.0, 1.0]) / np.sqrt(2), 0.4 / np.sqrt(2)),
        (np.array([0.0, 1.0, 1.0]) / np.sqrt(2), 0.0),
        (np.array([-1.0, 1.0, 1.0]) / np.sqrt(3), -0.4 / np.sqrt(3)),
    ]
    truth = np.array(baseline["true_parameters"])
    shift = np.array([0.001, -0.001, 0.002])
    offsets = {
        "none": np.zeros(4),
        "first_three_inconsistent": np.array([0.001, -0.001, 0.001, 0.0]),
        "rigid_layout_shift": np.array([n @ shift for n, d in planes]),
    }
    for delta in (-0.006, -0.001, 0.001, 0.006):
        offsets[f"fourth_only:{delta:+.3f}"] = np.array([0.0, 0.0, 0.0, delta])
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
                perturbation = np.random.default_rng(seed).uniform(-noise, noise, (4, len(rays)))
                for label, offset in offsets.items():
                    actual = [(n, d + v) for (n, d), v in zip(planes, offset, strict=True)]
                    observed = predict(rays, actual, truth) + perturbation
                    heldout = check_redundant_reference(rays, planes, observed)
                    joint = fit_reference(rays, planes, observed)
                    residual = observed - predict(rays, planes, joint["parameters"])
                    joint_p95 = np.quantile(np.abs(residual), 0.95, axis=1)
                    joint["per_plane_p95_m"] = joint_p95.tolist()
                    joint["diagnostic_status"] = (
                        "unknown"
                        if joint["status"] != "converged"
                        else "alarm"
                        if max(joint_p95) > 0.003
                        else "no_alarm"
                    )
                    for result in (heldout["fit"], joint):
                        error = np.array(result["parameters"]) - truth
                        result["scoring_only"] = {
                            "translation_error_norm_m": float(np.linalg.norm(error[:3])),
                            "depth_bias_error_m": float(error[6]),
                        }
                    cases.append(
                        {
                            "patch": patch,
                            "noise_bound_m": noise,
                            "seed": seed,
                            "plane_error": label,
                            "plane_offset_errors_m": offset.tolist(),
                            "heldout": heldout,
                            "joint": joint,
                        }
                    )
    summary = {}
    for label in offsets:
        rows = [c for c in cases if c["plane_error"] == label]
        summary[label] = {
            "cases": len(rows),
            "heldout": {
                s: sum(c["heldout"]["status"] == s for c in rows)
                for s in ("alarm", "no_alarm", "unknown")
            },
            "joint": {
                s: sum(c["joint"]["diagnostic_status"] == s for c in rows)
                for s in ("alarm", "no_alarm", "unknown")
            },
        }
    report = {
        "status": "complete",
        "experiment": "synthetic_redundant_reference",
        "script_sha256": digest(__file__),
        "implementation_sha256": digest("src/embodied_agent/redundant_reference.py"),
        "baseline_sha256": digest(baseline_path),
        "summary": summary,
        "cases": cases,
        "planes": [{"normal": n.tolist(), "offset_m": d} for n, d in planes],
        "true_parameters": truth.tolist(),
        "parameters": {
            "samples_per_plane": 441,
            "threshold_m": 0.003,
            "seeds": [73, 91, 117],
            "patch_half_widths": {
                "wide": [0.25, 0.20],
                "medium": [0.08, 0.06],
                "narrow": [0.025, 0.02],
            },
        },
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "calibration_certified": False,
        "limitations": [
            "ideal known correspondences in separate plane exposures; no rendered occlusion",
            "heldout fourth plane never enters first-three fit; joint residual is not heldout evidence",
            "unknown fits retained rather than counted as no alarm",
            "3mm threshold is diagnostic and not calibrated to a false alarm probability",
            "rigid reference-frame translation remains indistinguishable from camera translation",
            "no online calibration applied",
        ],
    }
    Path("docs/evaluations/redundant-reference-v1.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
