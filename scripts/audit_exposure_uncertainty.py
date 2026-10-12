"""Fresh paired calibration/validation batches; fixed uncertainty-aware rule."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from embodied_agent.exposure_uncertainty import bound_repeated_noise
from embodied_agent.memory_policy import digest
from embodied_agent.reference_fit import fit_reference, predict
from embodied_agent.reference_statistics import mean_prediction_uncertainty
from scripts.audit_exposure_batches import (
    SIGMA_KEYS,
    VARIANCE_KEYS,
    counts,
    distribution,
    draw,
    thresholds,
)

RULES = ("plugin", "bounded", "oracle")


def seed_plan(config):
    ranges = []
    for start, size in (
        (config["calibration_seed_start"], config["repeats_per_batch"]),
        (config["validation_seed_start"], config["validation_per_batch"]),
    ):
        if size < 1 or config["batch_count"] < 2:
            raise ValueError("positive batch sizes and at least two batches required")
        ranges.append(
            [
                list(range(start + b * size, start + (b + 1) * size))
                for b in range(config["batch_count"])
            ]
        )
    flat = [seed for group in ranges for batch in group for seed in batch]
    if len(flat) != len(set(flat)):
        raise ValueError("all calibration and validation batch seeds must be disjoint")
    return ranges


def summarize(batch_decisions, offsets, bootstrap_indices):
    """Resample whole batches, retaining pairing and shared calibration dependence."""
    a = np.asarray(batch_decisions)  # batch, case, rule, offset
    results = []
    for k, offset in enumerate(offsets):
        alarms = (a[:, :, :, k] == 1).sum(axis=1)
        known = (a[:, :, :, k] != -1).sum(axis=1)
        sampled_known = known[bootstrap_indices].sum(axis=1)
        sampled_alarms = alarms[bootstrap_indices].sum(axis=1)
        rates = np.divide(
            sampled_alarms,
            sampled_known,
            out=np.full(sampled_known.shape, np.nan),
            where=sampled_known > 0,
        )
        rows = {}
        for j, rule in enumerate(RULES):
            pooled = counts(a[:, :, j, k].ravel())
            # Ordinary pooled Wilson intervals would ignore shared calibration.
            pooled.pop("wilson_95_percent_converged")
            valid = rates[:, j][np.isfinite(rates[:, j])]
            rows[rule] = {
                **pooled,
                "cluster_bootstrap_95_percent_converged": np.quantile(
                    valid, [0.025, 0.975]
                ).tolist()
                if valid.size
                else None,
                "batch_alarm_fraction_converged": distribution(
                    [alarm / n for alarm, n in zip(alarms[:, j], known[:, j]) if n]
                ),
            }
        delta = rates[:, 1] - rates[:, 0]
        valid = delta[np.isfinite(delta)]
        results.append(
            {
                "offset_m": offset,
                "rules": rows,
                "bounded_minus_plugin_paired_cluster_95_percent": np.quantile(
                    valid, [0.025, 0.975]
                ).tolist()
                if valid.size
                else None,
                "plugin_alarm_removed_by_bound": int(
                    np.sum((a[:, :, 0, k] == 1) & (a[:, :, 1, k] == 0))
                ),
                "bound_only_alarm": int(np.sum((a[:, :, 0, k] == 0) & (a[:, :, 1, k] == 1))),
            }
        )
    return results


def main():
    config_path = Path("config/exposure-uncertainty.json")
    config = json.loads(config_path.read_text())
    calibration, validation = seed_plan(config)
    if not np.isclose(
        config["nominal_alpha"],
        config["calibration_failure_probability"] + config["bounded_rule_tail_probability"],
    ):
        raise ValueError("calibration and diagnostic error budgets must sum to nominal alpha")
    bootstrap_indices = np.random.default_rng(config["cluster_bootstrap_seed"]).integers(
        0, config["batch_count"], (config["cluster_bootstrap_replicates"], config["batch_count"])
    )
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
                predict(rays, planes[:3] + [(planes[3][0], planes[3][1] + s)], truth)[3] - clean[3]
                for s in offsets
            ]
        )
        for model in ("independent", "global_shared", "exposure_shared"):
            e = 0.001 if model == "exposure_shared" else 0.0
            g = 0.001 if model == "global_shared" else 0.0
            batches, batch_decisions = [], []
            for batch_id, (seeds, validation_seeds) in enumerate(zip(calibration, validation)):
                observations = np.array([draw(clean, seed, e, g) for seed in seeds])
                bound = bound_repeated_noise(
                    observations, config["calibration_failure_probability"]
                )
                sigmas = [
                    [bound["estimate"][key] for key in SIGMA_KEYS],
                    bound["upper_sigmas_m"],
                    [0.001, e, g],
                ]
                decisions = np.full((len(validation_seeds), 3, len(offsets)), -1, dtype=int)
                cases, threshold_ratios = [], []
                for i, seed in enumerate(validation_seeds):
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
                        unit = mean_prediction_uncertainty(rays, planes, fit["parameters"], 1, 1, 1)
                        coefficients = [
                            unit["variance_components_m2"][key] for key in VARIANCE_KEYS
                        ]
                        limits = [
                            float(thresholds(coefficients, s, alpha))
                            for s, alpha in zip(
                                sigmas,
                                [
                                    config["nominal_alpha"],
                                    config["bounded_rule_tail_probability"],
                                    config["nominal_alpha"],
                                ],
                            )
                        ]
                        decisions[i] = (abs(residuals)[None, :] > np.array(limits)[:, None]).astype(
                            int
                        )
                        case.update({"variance_coefficients": coefficients, "thresholds_m": limits})
                        threshold_ratios.append(limits[1] / limits[0])
                    case["decisions"] = decisions[i].tolist()
                    cases.append(case)
                batch_decisions.append(decisions)
                batches.append(
                    {
                        "batch_id": batch_id,
                        "calibration_seed_range_inclusive": [seeds[0], seeds[-1]],
                        "validation_seed_range_inclusive": [
                            validation_seeds[0],
                            validation_seeds[-1],
                        ],
                        "observations_sha256": hashlib.sha256(observations.tobytes()).hexdigest(),
                        "bound": bound,
                        "scoring_only_true_variances_covered": bool(
                            np.all(
                                np.array(bound["upper_variance_components_m2"])
                                >= np.array([0.001, e, g]) ** 2
                            )
                        ),
                        "fit_status_counts": dict(Counter(c["fit_status"] for c in cases)),
                        "bounded_to_plugin_threshold_ratio": distribution(threshold_ratios),
                        "cases": cases,
                        "summary": [
                            {
                                "offset_m": shift,
                                "rules": {
                                    rule: counts(decisions[:, j, k]) for j, rule in enumerate(RULES)
                                },
                            }
                            for k, shift in enumerate(offsets)
                        ],
                    }
                )
                if (batch_id + 1) % 8 == 0:
                    print(patch, model, f"{batch_id + 1}/{len(calibration)} batches", flush=True)
            summary = summarize(batch_decisions, offsets, bootstrap_indices)
            conditions.append(
                {
                    "patch": patch,
                    "noise_model": model,
                    "batches": batches,
                    "summary": summary,
                    "scoring_only_true_sigmas_m": [0.001, e, g],
                    "fit_status_counts": dict(
                        sum((Counter(b["fit_status_counts"]) for b in batches), Counter())
                    ),
                    "scoring_only_variance_coverage_batches": sum(
                        b["scoring_only_true_variances_covered"] for b in batches
                    ),
                }
            )
            print(
                patch,
                model,
                "null counts",
                {rule: summary[0]["rules"][rule]["alarms"] for rule in RULES},
                flush=True,
            )
    paths = [
        Path(__file__),
        config_path,
        baseline_path,
        Path("scripts/audit_exposure_batches.py"),
        *[
            Path(f"src/embodied_agent/{name}.py")
            for name in (
                "exposure_uncertainty",
                "repeated_exposure",
                "reference_fit",
                "reference_statistics",
                "multiplane_reference",
            )
        ],
    ]
    report = {
        "status": "complete",
        "experiment": "bounded_noise_uncertainty_independent_panels",
        "config": config,
        "conditions": conditions,
        "rule_order": list(RULES),
        "sha256": {str(path): digest(path) for path in paths},
        "decision_encoding": {"unknown": -1, "no_alarm": 0, "alarm": 1},
        "training_performed": False,
        "policy_test_executed": False,
        "recovery_executed": False,
        "correction_applied": False,
        "calibration_certified": False,
        "limitations": [
            "Gaussian homoscedastic stationary repeats with known pixel correspondence",
            "variance bounds use calibration only; pointwise total error budget is 1% only for fixed linear Gaussian contrast",
            "fitted weights, nonlinear residuals and conditioning on convergence prevent strict diagnostic coverage claim",
            "independent calibration/validation batches, but conditions and offsets remain paired",
            "bootstrap resamples 32 whole batches; percentile intervals are approximate and rare-event tails limited",
            "conservative variance bounds and split error budget may reduce detection power",
            "global constant absorbed by bias; unknown narrow fits excluded from conditional rates and also reported separately",
        ],
        "method_source": "https://doi.org/10.1214/aos/1015957395",
    }
    Path("docs/evaluations/exposure-uncertainty-v1.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
