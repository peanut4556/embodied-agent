"""Prespecified new Monte Carlo seeds; no threshold selection on evaluation data."""

import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.reference_fit import fit_reference, predict
from embodied_agent.reference_statistics import mean_prediction_uncertainty, wilson_interval
from embodied_agent.repeated_exposure import estimate_repeated_noise


def main():
    baseline_path = Path("docs/evaluations/redundant-reference-v1.json")
    baseline = json.loads(baseline_path.read_text())
    planes = [(np.array(p["normal"]), p["offset_m"]) for p in baseline["planes"]]
    truth = np.array(baseline["true_parameters"])
    cases = []
    estimates = []
    for patch, (ex, ey) in (("wide", (0.25, 0.20)), ("narrow", (0.025, 0.02))):
        rays = np.array(
            [[x, y, -1.0] for x in np.linspace(-ex, ex, 21) for y in np.linspace(-ey, ey, 21)]
        )
        clean = predict(rays, planes, truth)
        for noise_model in ("independent", "global_shared", "exposure_shared"):
            exposure_sigma = 0.001 if noise_model == "exposure_shared" else 0.0
            global_sigma = 0.001 if noise_model == "global_shared" else 0.0
            repeated = []
            for calibration_seed in range(2000, 2128):
                calibration_rng = np.random.default_rng(calibration_seed)
                batch = (
                    clean
                    + calibration_rng.normal(0, 0.001, clean.shape)
                    + calibration_rng.normal(0, 1, (4, 1)) * exposure_sigma
                    + calibration_rng.normal(0, 1) * global_sigma
                )
                repeated.append(batch)
            repeated = np.array(repeated)
            estimated = estimate_repeated_noise(repeated)
            estimates.append(
                {
                    "patch": patch,
                    "noise_model": noise_model,
                    "estimate": estimated,
                    "observations_sha256": hashlib.sha256(repeated.tobytes()).hexdigest(),
                    "scoring_only_true_sigmas_m": [0.001, exposure_sigma, global_sigma],
                }
            )
            for seed in range(3000, 3080):
                rng = np.random.default_rng(seed)
                # Draw all components regardless of mode to pair comparisons.
                pixel = rng.normal(0, 0.001, clean.shape)
                exposure = rng.normal(0, 1, (4, 1)) * exposure_sigma
                shared = rng.normal(0, 1) * global_sigma
                observed = clean + pixel + exposure + shared
                fit = fit_reference(rays, planes[:3], observed[:3])
                thresholds = {}
                if fit["status"] == "converged":
                    for rule, pixel_sigma, e, g in (
                        (
                            "estimated",
                            estimated["sigma_pixel_m"],
                            estimated["sigma_exposure_m"],
                            estimated["sigma_global_m"],
                        ),
                        ("specified_correlation", 0.001, exposure_sigma, global_sigma),
                    ):
                        thresholds[rule] = mean_prediction_uncertainty(
                            rays, planes, fit["parameters"], pixel_sigma, e, g
                        )
                for shift in (0.0, -0.001, 0.001, 0.006):
                    changed_planes = planes[:3] + [(planes[3][0], planes[3][1] + shift)]
                    fourth = (
                        predict(rays, changed_planes, truth)[3] + pixel[3] + exposure[3] + shared
                    )
                    mean = float(np.mean(fourth - predict(rays, planes[3:], fit["parameters"])[0]))
                    decisions = {
                        rule: {
                            "status": "unknown"
                            if rule not in thresholds
                            else "alarm"
                            if abs(mean) > thresholds[rule]["two_sided_99_percent_threshold_m"]
                            else "no_alarm",
                            "uncertainty": thresholds.get(rule),
                        }
                        for rule in ("estimated", "specified_correlation")
                    }
                    cases.append(
                        {
                            "patch": patch,
                            "noise_model": noise_model,
                            "seed": seed,
                            "fourth_plane_offset_m": shift,
                            "fit_status": fit["status"],
                            "heldout_mean_residual_m": mean,
                            "decisions": decisions,
                        }
                    )
    summary = []
    for patch in ("wide", "narrow"):
        for model in ("independent", "global_shared", "exposure_shared"):
            for shift in (0.0, -0.001, 0.001, 0.006):
                for rule in ("estimated", "specified_correlation"):
                    rows = [
                        c["decisions"][rule]["status"]
                        for c in cases
                        if c["patch"] == patch
                        and c["noise_model"] == model
                        and c["fourth_plane_offset_m"] == shift
                    ]
                    alarms, unknown = rows.count("alarm"), rows.count("unknown")
                    known = len(rows) - unknown
                    summary.append(
                        {
                            "patch": patch,
                            "noise_model": model,
                            "offset_m": shift,
                            "rule": rule,
                            "cases": len(rows),
                            "alarms": alarms,
                            "unknown": unknown,
                            "alarm_fraction_all_cases": alarms / len(rows),
                            "alarm_fraction_converged": alarms / known if known else None,
                            "wilson_95_percent_converged": wilson_interval(alarms, known),
                        }
                    )
    report = {
        "status": "complete",
        "experiment": "synthetic_repeated_exposure_estimation",
        "script_sha256": digest(__file__),
        "implementation_sha256": digest("src/embodied_agent/repeated_exposure.py"),
        "statistics_sha256": digest("src/embodied_agent/reference_statistics.py"),
        "noise_estimates": estimates,
        "fit_sha256": digest("src/embodied_agent/reference_fit.py"),
        "geometry_sha256": digest("src/embodied_agent/multiplane_reference.py"),
        "baseline_sha256": digest(baseline_path),
        "cases": cases,
        "summary": summary,
        "parameters": {
            "calibration_seeds": [2000, 2127],
            "validation_seeds": [3000, 3079],
            "seed_range_inclusive": True,
            "samples_per_plane": 441,
            "sigma_pixel_m": 0.001,
            "sigma_shared_m": 0.001,
            "two_sided_nominal_alpha": 0.01,
            "fourth_plane_offsets_m": [0.0, -0.001, 0.001, 0.006],
            "estimated_rule_uses_calibration_observations_only": True,
        },
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "calibration_certified": False,
        "limitations": [
            "Gaussian noise with specified covariance; differs from earlier bounded uniform noise",
            "estimated rule uses disjoint repeated observations; specified-correlation is an oracle comparison only",
            "stationary scene and correspondence assumed; negative variance components clipped and recorded",
            "single calibration batch per condition; variance-estimation uncertainty not included in threshold",
            "first-order mean statistic can miss zero-mean spatial defects",
            "pointwise nominal 1% test, not family-wise error control across all trials",
            "rates conditional on convergence have selection effects; unknowns reported separately",
            "paired seeds across conditions; confidence intervals apply within a condition only",
            "global shared depth constant is representable by fitted bias and may remain invisible",
            "no real-camera calibration or online policy change",
        ],
    }
    Path("docs/evaluations/repeated-exposure-v1.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for row in summary:
        if row["offset_m"] in (0.0, 0.001):
            print(row)


if __name__ == "__main__":
    main()
