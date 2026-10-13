"""Independent new batches isolate tail-budget splitting and variance-bound changes."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from embodied_agent.exposure_chernoff import bound_repeated_noise_chernoff
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
from scripts.audit_exposure_uncertainty import seed_plan

RULES = ("plugin", "split_only", "lm", "chernoff", "oracle")
PAIRS = (
    ("split_only", "plugin"),
    ("lm", "split_only"),
    ("chernoff", "split_only"),
    ("chernoff", "lm"),
)


def summarize(decisions, offsets, indices):
    a = np.asarray(decisions)
    results = []
    for k, shift in enumerate(offsets):
        alarms = (a[:, :, :, k] == 1).sum(1)
        known = (a[:, :, :, k] != -1).sum(1)
        denominator = known[indices].sum(1)
        rates = np.divide(
            alarms[indices].sum(1),
            denominator,
            out=np.full(denominator.shape, np.nan),
            where=denominator > 0,
        )
        rules, contrasts = {}, {}
        for j, rule in enumerate(RULES):
            row = counts(a[:, :, j, k].ravel())
            row.pop("wilson_95_percent_converged")
            valid = rates[:, j][np.isfinite(rates[:, j])]
            rules[rule] = {
                **row,
                "cluster_95_percent": np.quantile(valid, [0.025, 0.975]).tolist()
                if valid.size
                else None,
                "batch_rate_distribution": distribution(
                    [n / d for n, d in zip(alarms[:, j], known[:, j]) if d]
                ),
            }
        for left, right in PAIRS:
            li, ri = RULES.index(left), RULES.index(right)
            delta = rates[:, li] - rates[:, ri]
            valid = delta[np.isfinite(delta)]
            contrasts[f"{left}_minus_{right}"] = {
                "paired_cluster_95_percent": np.quantile(valid, [0.025, 0.975]).tolist()
                if valid.size
                else None,
                "alarms_added": int(np.sum((a[:, :, li, k] == 1) & (a[:, :, ri, k] == 0))),
                "alarms_removed": int(np.sum((a[:, :, li, k] == 0) & (a[:, :, ri, k] == 1))),
            }
        results.append({"offset_m": shift, "rules": rules, "contrasts": contrasts})
    return results


def rule_inputs(bounds, true_sigmas, config):
    estimate = [bounds["calibration"]["estimate"][key] for key in SIGMA_KEYS]
    return (
        [
            estimate,
            estimate,
            bounds["calibration"]["upper_sigmas_m"],
            bounds["chernoff"]["upper_sigmas_m"],
            true_sigmas,
        ],
        [
            config["nominal_alpha"],
            *[config["bounded_rule_tail_probability"]] * 3,
            config["nominal_alpha"],
        ],
    )


def main():
    config_path = Path("config/exposure-ablation.json")
    config = json.loads(config_path.read_text())
    calibration, validation = seed_plan(config)
    assert config["rule_order"] == list(RULES)
    assert np.isclose(
        config["nominal_alpha"],
        config["calibration_failure_probability"] + config["bounded_rule_tail_probability"],
    )
    indices = np.random.default_rng(config["cluster_bootstrap_seed"]).integers(
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
            batches, matrices = [], []
            for batch_id, (seeds, panel) in enumerate(zip(calibration, validation)):
                observations = np.array([draw(clean, seed, e, g) for seed in seeds])
                bounds = bound_repeated_noise_chernoff(
                    observations, config["calibration_failure_probability"]
                )
                sigmas, alphas = rule_inputs(bounds, [0.001, e, g], config)
                decisions = np.full((len(panel), len(RULES), len(offsets)), -1, dtype=int)
                cases, ratios = [], []
                for i, seed in enumerate(panel):
                    observed = draw(clean, seed, e, g)
                    fit = fit_reference(rays, planes[:3], observed[:3])
                    residuals = (
                        observed[3] + shifts - predict(rays, planes[3:], fit["parameters"])[0]
                    ).mean(1)
                    case = {
                        "seed": seed,
                        "fit_status": fit["status"],
                        "residuals_m": residuals.tolist(),
                    }
                    if fit["status"] == "converged":
                        unit = mean_prediction_uncertainty(rays, planes, fit["parameters"], 1, 1, 1)
                        coefficients = [
                            unit["variance_components_m2"][key] for key in VARIANCE_KEYS
                        ]
                        limits = np.array(
                            [
                                float(thresholds(coefficients, s, alpha))
                                for s, alpha in zip(sigmas, alphas)
                            ]
                        )
                        decisions[i] = (abs(residuals)[None, :] > limits[:, None]).astype(int)
                        case.update(
                            {"variance_coefficients": coefficients, "thresholds_m": limits.tolist()}
                        )
                        ratios.append(limits / limits[0])
                    case["decisions"] = decisions[i].tolist()
                    cases.append(case)
                matrices.append(decisions)
                batches.append(
                    {
                        "batch_id": batch_id,
                        "bounds": bounds,
                        "calibration_seeds_inclusive": [seeds[0], seeds[-1]],
                        "validation_seeds_inclusive": [panel[0], panel[-1]],
                        "observations_sha256": hashlib.sha256(observations.tobytes()).hexdigest(),
                        "scoring_only_coverage": {
                            name: bool(
                                np.all(
                                    np.array(bounds[name]["upper_variance_components_m2"])
                                    >= np.array([0.001, e, g]) ** 2
                                )
                            )
                            for name in ("calibration", "chernoff")
                        },
                        "fit_status_counts": dict(Counter(c["fit_status"] for c in cases)),
                        "threshold_ratio_to_plugin": {
                            rule: distribution(np.array(ratios)[:, j])
                            for j, rule in enumerate(RULES)
                        }
                        if ratios
                        else {},
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
                    print(patch, model, f"{batch_id + 1}/32 batches", flush=True)
            summary = summarize(matrices, offsets, indices)
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
                }
            )
            print(
                patch,
                model,
                "null",
                {r: summary[0]["rules"][r]["alarms"] for r in RULES},
                flush=True,
            )
    paths = [
        Path(__file__),
        config_path,
        baseline_path,
        *[
            Path(f"scripts/{name}.py")
            for name in ("audit_exposure_batches", "audit_exposure_uncertainty")
        ],
        *[
            Path(f"src/embodied_agent/{name}.py")
            for name in (
                "exposure_chernoff",
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
        "experiment": "variance_bound_and_tail_budget_ablation",
        "config": config,
        "rule_order": list(RULES),
        "conditions": conditions,
        "sha256": {str(p): digest(p) for p in paths},
        "decision_encoding": {"unknown": -1, "no_alarm": 0, "alarm": 1},
        "training_performed": False,
        "policy_test_executed": False,
        "recovery_executed": False,
        "correction_applied": False,
        "calibration_certified": False,
        "limitations": [
            "fixed Gaussian stationary noise model; Chernoff bounds are not exact chi-square quantiles",
            "split_only is an ablation control, not a variance coverage guarantee",
            "fitted nonlinear weights and convergence selection prevent strict diagnostic error guarantee",
            "32 independent batches per condition; conditions and offsets paired, do not pool as independent trials",
            "cluster percentile bootstrap intervals approximate; rare-event and multiple-comparison limitations",
            "unknown fits preserved; global static bias may remain invisible",
        ],
    }
    Path("docs/evaluations/exposure-ablation-v1.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
