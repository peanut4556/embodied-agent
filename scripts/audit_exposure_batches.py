"""Independent calibration batches crossed with one untouched validation panel."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import NormalDist

import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.reference_fit import fit_reference, predict
from embodied_agent.reference_statistics import mean_prediction_uncertainty, wilson_interval
from embodied_agent.repeated_exposure import estimate_repeated_noise

SIGMA_KEYS = ("sigma_pixel_m", "sigma_exposure_m", "sigma_global_m")
VARIANCE_KEYS = ("independent_pixels", "shared_within_exposure", "shared_across_all_exposures")


def seed_plan(config):
    batches = [
        list(
            range(
                config["calibration_seed_start"] + i * config["repeats_per_batch"],
                config["calibration_seed_start"] + (i + 1) * config["repeats_per_batch"],
            )
        )
        for i in range(config["batch_count"])
    ]
    validation = list(
        range(
            config["validation_seed_start"],
            config["validation_seed_start"] + config["validation_count"],
        )
    )
    flat = [seed for batch in batches for seed in batch]
    if not batches or not validation or any(len(batch) < 2 for batch in batches):
        raise ValueError("nonempty batches with at least two repeats and validation required")
    if len(set(flat)) != len(flat) or set(flat) & set(validation):
        raise ValueError("calibration batches and validation seeds must be disjoint")
    return batches, validation


def distribution(values):
    x = np.asarray(values, dtype=float)
    if not x.size:
        return None
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "sample_sd": float(x.std(ddof=1)) if x.size > 1 else None,
        "min": float(x.min()),
        "p025": float(np.quantile(x, 0.025)),
        "median": float(np.median(x)),
        "p975": float(np.quantile(x, 0.975)),
        "max": float(x.max()),
    }


def thresholds(coefficients, sigmas, alpha=0.01):
    return np.maximum(
        1e-9,
        NormalDist().inv_cdf(1 - alpha / 2)
        * np.sqrt(np.asarray(sigmas) ** 2 @ np.asarray(coefficients)),
    )


def counts(decisions):
    decisions = np.asarray(decisions)
    alarm, unknown = int(np.sum(decisions == 1)), int(np.sum(decisions == -1))
    total = int(decisions.size)
    known = total - unknown
    return {
        "cases": total,
        "alarms": alarm,
        "unknown": unknown,
        "alarm_fraction_all_cases": alarm / total if total else None,
        "alarm_fraction_converged": alarm / known if known else None,
        "wilson_95_percent_converged": wilson_interval(alarm, known),
    }


def draw(clean, seed, exposure_sigma, global_sigma):
    rng = np.random.default_rng(seed)
    return (
        clean
        + rng.normal(0, 0.001, clean.shape)
        + rng.normal(0, 1, (4, 1)) * exposure_sigma
        + rng.normal(0, 1) * global_sigma
    )


def main():
    config_path = Path("config/repeated-exposure-batches.json")
    config = json.loads(config_path.read_text())
    batches, validation = seed_plan(config)
    baseline_path = Path("docs/evaluations/redundant-reference-v1.json")
    baseline = json.loads(baseline_path.read_text())
    planes = [(np.array(p["normal"]), p["offset_m"]) for p in baseline["planes"]]
    truth = np.array(baseline["true_parameters"])
    offsets = config["fourth_plane_offsets_m"]
    conditions = []
    for patch, (ex, ey) in (("wide", (0.25, 0.20)), ("narrow", (0.025, 0.02))):
        rays = np.array(
            [[x, y, -1.0] for x in np.linspace(-ex, ex, 21) for y in np.linspace(-ey, ey, 21)]
        )
        clean = predict(rays, planes, truth)
        shifts = np.array(
            [
                predict(rays, planes[:3] + [(planes[3][0], planes[3][1] + shift)], truth)[3]
                - clean[3]
                for shift in offsets
            ]
        )
        for model in ("independent", "global_shared", "exposure_shared"):
            e, g = (
                0.001 if model == "exposure_shared" else 0.0,
                0.001 if model == "global_shared" else 0.0,
            )
            estimates = []
            for batch_id, seeds in enumerate(batches):
                observed = np.array([draw(clean, seed, e, g) for seed in seeds])
                estimates.append(
                    {
                        "batch_id": batch_id,
                        "seed_range_inclusive": [seeds[0], seeds[-1]],
                        "observations_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                        "estimate": estimate_repeated_noise(observed),
                    }
                )
            sigmas = np.array([[row["estimate"][key] for key in SIGMA_KEYS] for row in estimates])
            decisions = np.full((len(batches), len(validation), len(offsets)), -1, dtype=int)
            oracle = np.full((len(validation), len(offsets)), -1, dtype=int)
            cases, ratios = [], []
            for index, seed in enumerate(validation):
                observed = draw(clean, seed, e, g)
                fit = fit_reference(rays, planes[:3], observed[:3])
                residuals = (
                    observed[3] + shifts - predict(rays, planes[3:], fit["parameters"])[0]
                ).mean(axis=1)
                case = {
                    "seed": seed,
                    "fit_status": fit["status"],
                    "heldout_mean_residuals_m": residuals.tolist(),
                }
                if fit["status"] == "converged":
                    # Variance is linear in squared noise scales. One Jacobian per fit
                    # suffices for all calibration batches; tested against direct calls.
                    unit = mean_prediction_uncertainty(rays, planes, fit["parameters"], 1, 1, 1)
                    coefficients = [unit["variance_components_m2"][key] for key in VARIANCE_KEYS]
                    limits = thresholds(coefficients, sigmas, config["two_sided_nominal_alpha"])
                    known_limit = float(
                        thresholds(coefficients, [0.001, e, g], config["two_sided_nominal_alpha"])
                    )
                    decisions[:, index, :] = (abs(residuals)[None, :] > limits[:, None]).astype(int)
                    oracle[index] = (abs(residuals) > known_limit).astype(int)
                    ratios.append(limits / known_limit)
                    case.update(
                        {"variance_coefficients": coefficients, "oracle_threshold_m": known_limit}
                    )
                case.update(
                    {
                        "estimated_alarm_batch_counts": (decisions[:, index] == 1).sum(0).tolist(),
                        "oracle_decisions": oracle[index].tolist(),
                    }
                )
                cases.append(case)
            batch_summary, summary = [], []
            for batch_id in range(len(batches)):
                batch_summary.append(
                    {
                        "batch_id": batch_id,
                        "offsets": [
                            {
                                "offset_m": shift,
                                **counts(decisions[batch_id, :, k]),
                                "oracle_disagreements": int(
                                    np.sum(decisions[batch_id, :, k] != oracle[:, k])
                                ),
                            }
                            for k, shift in enumerate(offsets)
                        ],
                    }
                )
            for k, shift in enumerate(offsets):
                rows = [batch["offsets"][k] for batch in batch_summary]
                summary.append(
                    {
                        "offset_m": shift,
                        "oracle": counts(oracle[:, k]),
                        "estimated_alarm_fraction_converged_across_batches": distribution(
                            [
                                row["alarm_fraction_converged"]
                                for row in rows
                                if row["alarm_fraction_converged"] is not None
                            ]
                        ),
                        "estimated_alarms_across_batches": distribution(
                            [row["alarms"] for row in rows]
                        ),
                        "oracle_disagreements_across_batches": distribution(
                            [row["oracle_disagreements"] for row in rows]
                        ),
                        "validation_cases_changing_decision_across_batches": int(
                            np.sum(decisions[:, :, k].min(0) != decisions[:, :, k].max(0))
                        ),
                    }
                )
            conditions.append(
                {
                    "patch": patch,
                    "noise_model": model,
                    "estimates": estimates,
                    "scoring_only_true_sigmas_m": [0.001, e, g],
                    "sigma_distributions_m": {
                        key: distribution(sigmas[:, i]) for i, key in enumerate(SIGMA_KEYS)
                    },
                    "clipped_batch_counts": {
                        key: sum(
                            key in row["estimate"]["negative_components_clipped"]
                            for row in estimates
                        )
                        for key in ("exposure", "global")
                    },
                    "fit_status_counts": dict(Counter(case["fit_status"] for case in cases)),
                    "threshold_ratio_to_oracle_per_batch": [
                        distribution(np.array(ratios)[:, i]) for i in range(len(batches))
                    ]
                    if ratios
                    else [],
                    "cases": cases,
                    "batch_summary": batch_summary,
                    "summary": summary,
                }
            )
            print(
                patch,
                model,
                conditions[-1]["fit_status_counts"],
                "null alarms",
                summary[0]["estimated_alarms_across_batches"],
                flush=True,
            )
    report = {
        "status": "complete",
        "experiment": "independent_repeated_exposure_batches",
        "config": config,
        "conditions": conditions,
        "sha256": {
            str(path): digest(path)
            for path in (
                Path(__file__),
                config_path,
                baseline_path,
                Path("src/embodied_agent/repeated_exposure.py"),
                Path("src/embodied_agent/reference_statistics.py"),
                Path("src/embodied_agent/reference_fit.py"),
                Path("src/embodied_agent/multiplane_reference.py"),
            )
        },
        "decision_encoding": {"unknown": -1, "no_alarm": 0, "alarm": 1},
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "calibration_certified": False,
        "limitations": [
            "32 independent calibration batches, paired seeds across conditions",
            "all batches share one disjoint validation panel; do not pool batch-case counts as independent trials",
            "batch quantiles describe dispersion, not confidence intervals or population coverage guarantees",
            "Wilson intervals are within one batch and condition, conditional on fit convergence",
            "validation sampling variability is not isolated by independent validation panels",
            "fixed plug-in nominal 1% thresholds; estimation uncertainty not propagated or tuned",
            "synthetic stationary Gaussian exposures; no real-camera or static-bias certification",
            "global shared constant can be absorbed by fitted bias; narrow fit failures remain unknown",
        ],
    }
    Path("docs/evaluations/repeated-exposure-batches-v1.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
