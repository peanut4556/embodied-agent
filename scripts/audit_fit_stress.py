"""Stress the fixed candidate stop without tuning it or pretending stationarity certifies a model."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from embodied_agent.fit_stress import stress_geometry, stress_observation
from embodied_agent.memory_policy import digest
from embodied_agent.reference_fit import fit_reference, predict
from embodied_agent.reference_fit_candidate import USABLE, fit_reference_candidate
from embodied_agent.reference_statistics import mean_prediction_uncertainty
from embodied_agent.repeated_exposure import estimate_repeated_noise
from scripts.audit_exposure_batches import (
    SIGMA_KEYS,
    VARIANCE_KEYS,
    counts,
    distribution,
    thresholds,
)
from scripts.audit_exposure_uncertainty import seed_plan


def summarize(matrices, offsets, indices):
    a = np.asarray(matrices)
    rows = []
    for k, offset in enumerate(offsets):
        rules = {}
        for j, name in enumerate(("legacy", "candidate")):
            x = a[:, :, j, k]
            alarms, usable = (x == 1).sum(1), (x != -1).sum(1)
            denominator = usable[indices].sum(1)
            rates = np.divide(
                alarms[indices].sum(1),
                denominator,
                out=np.full(denominator.shape, np.nan),
                where=denominator > 0,
            )
            valid = rates[np.isfinite(rates)]
            row = counts(x.ravel())
            row.pop("wilson_95_percent_converged")
            row["alarm_fraction_usable"] = row.pop("alarm_fraction_converged")
            rules[name] = {
                **row,
                "cluster_95_percent": np.quantile(valid, [0.025, 0.975]).tolist()
                if valid.size
                else None,
            }
        common = (a[:, :, 0, k] != -1) & (a[:, :, 1, k] != -1)
        new = (a[:, :, 0, k] == -1) & (a[:, :, 1, k] != -1)
        rows.append(
            {
                "offset_m": offset,
                "rules": rules,
                "common_usable": int(common.sum()),
                "common_decision_disagreements": int(
                    np.sum(common & (a[:, :, 0, k] != a[:, :, 1, k]))
                ),
                "newly_usable": int(new.sum()),
                "newly_usable_candidate_alarms": int(np.sum(new & (a[:, :, 1, k] == 1))),
                "lost_usable": int(np.sum((a[:, :, 0, k] != -1) & (a[:, :, 1, k] == -1))),
            }
        )
    return rows


def risk_summary(cases, limit):
    rows = {}
    for j, name in enumerate(("legacy", "candidate")):
        usable = [c for c in cases if c["decisions"][j][0] != -1]
        clear = [c for c in usable if c["decisions"][j][0] == 0]
        risky = [c for c in clear if c["scoring_errors"][j]["translation_norm_m"] > limit]
        rows[name] = {
            "usable": len(usable),
            "nominal_fourth_no_alarm": len(clear),
            "no_alarm_translation_error_over_limit": len(risky),
            "translation_error_usable_m": distribution(
                [c["scoring_errors"][j]["translation_norm_m"] for c in usable]
            ),
            "newly_usable_no_alarm_translation_error_over_limit": sum(
                c["decisions"][0][0] == -1 for c in risky
            )
            if j == 1
            else 0,
        }
    return rows


def main():
    config_path = Path("config/fit-stress.json")
    config = json.loads(config_path.read_text())
    calibration, validation = seed_plan(config)
    indices = np.random.default_rng(config["cluster_bootstrap_seed"]).integers(
        0, config["batch_count"], (config["cluster_bootstrap_replicates"], config["batch_count"])
    )
    previous_path = Path("docs/evaluations/fit-stop-validation-v1.json")
    previous = json.loads(previous_path.read_text())
    for path, expected in previous["sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"previous implementation changed: {path}")
    baseline_path = Path("docs/evaluations/redundant-reference-v1.json")
    baseline = json.loads(baseline_path.read_text())
    planes = [(np.array(p["normal"]), p["offset_m"]) for p in baseline["planes"]]
    truth = np.array(baseline["true_parameters"])
    offsets = config["fourth_plane_offsets_m"]
    conditions = []
    for patch, (ex, ey) in (("wide", (0.25, 0.20)), ("narrow", (0.025, 0.02))):
        base_rays = np.array(
            [[x, y, -1.0] for x in np.linspace(-ex, ex, 21) for y in np.linspace(-ey, ey, 21)]
        )
        for scenario in config["scenarios"]:
            rays, actual_planes = stress_geometry(base_rays, planes, scenario, config)
            clean = predict(rays, actual_planes, truth)
            shifts = np.array(
                [
                    predict(
                        rays,
                        actual_planes[:3] + [(actual_planes[3][0], actual_planes[3][1] + s)],
                        truth,
                    )[3]
                    - clean[3]
                    for s in offsets
                ]
            )
            batches, matrices = [], []
            for bi, (seeds, panel) in enumerate(zip(calibration, validation)):
                repeated = np.array(
                    [
                        stress_observation(clean, rays, s, scenario, "calibration", config)
                        for s in seeds
                    ]
                )
                estimate = estimate_repeated_noise(repeated)
                sigmas = [estimate[key] for key in SIGMA_KEYS]
                decisions = np.full((len(panel), 2, len(offsets)), -1, dtype=int)
                cases = []
                for i, seed in enumerate(panel):
                    observed = stress_observation(clean, rays, seed, scenario, "validation", config)
                    fits = [
                        fit_reference(rays, planes[:3], observed[:3]),
                        fit_reference_candidate(rays, planes[:3], observed[:3]),
                    ]
                    case = {
                        "seed": seed,
                        "fits": fits,
                        "observations_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                        "diagnostics": [],
                        "scoring_errors": [],
                    }
                    predictions = []
                    for j, fit in enumerate(fits):
                        p = np.array(fit["parameters"])
                        fourth = predict(rays, planes[3:], p)[0]
                        predictions.append(fourth)
                        residual = (observed[3] + shifts - fourth).mean(1)
                        diagnostic = {"residuals_m": residual.tolist()}
                        usable = fit["status"] == "converged" if j == 0 else fit["status"] in USABLE
                        if usable:
                            unit = mean_prediction_uncertainty(rays, planes, p, 1, 1, 1)
                            coefs = [unit["variance_components_m2"][key] for key in VARIANCE_KEYS]
                            threshold = float(thresholds(coefs, sigmas, config["nominal_alpha"]))
                            decisions[i, j] = (abs(residual) > threshold).astype(int)
                            diagnostic.update(
                                {"variance_coefficients": coefs, "threshold_m": threshold}
                            )
                        case["diagnostics"].append(diagnostic)
                        case["scoring_errors"].append(
                            {
                                "translation_norm_m": float(np.linalg.norm(p[:3] - truth[:3])),
                                "rotation_vector_norm_rad": float(
                                    np.linalg.norm(p[3:6] - truth[3:6])
                                ),
                                "depth_bias_error_m": float(p[6] - truth[6]),
                            }
                        )
                    case.update(
                        {
                            "decisions": decisions[i].tolist(),
                            "max_heldout_prediction_delta_m": float(
                                np.max(abs(predictions[1] - predictions[0]))
                            ),
                        }
                    )
                    cases.append(case)
                matrices.append(decisions)
                batches.append(
                    {
                        "batch_id": bi,
                        "calibration_seeds_inclusive": [seeds[0], seeds[-1]],
                        "validation_seeds_inclusive": [panel[0], panel[-1]],
                        "estimate": estimate,
                        "calibration_observations_sha256": hashlib.sha256(
                            repeated.tobytes()
                        ).hexdigest(),
                        "cases": cases,
                    }
                )
                if (bi + 1) % 16 == 0:
                    print(patch, scenario, f"{bi + 1}/32 batches", flush=True)
            cases = [c for b in batches for c in b["cases"]]
            conditions.append(
                {
                    "patch": patch,
                    "scenario": scenario,
                    "batches": batches,
                    "summary": summarize(matrices, offsets, indices),
                    "fit_status_counts": {
                        name: dict(Counter(c["fits"][j]["status"] for c in cases))
                        for j, name in enumerate(("legacy", "candidate"))
                    },
                    "risk_summary": risk_summary(
                        cases, config["scoring_translation_error_limit_m"]
                    ),
                    "max_heldout_prediction_delta_m": max(
                        c["max_heldout_prediction_delta_m"] for c in cases
                    ),
                }
            )
            print(patch, scenario, conditions[-1]["fit_status_counts"], flush=True)
    paths = [
        Path(__file__),
        config_path,
        previous_path,
        baseline_path,
        *[
            Path(f"scripts/{n}.py")
            for n in ("audit_exposure_batches", "audit_exposure_uncertainty")
        ],
        *[
            Path(f"src/embodied_agent/{n}.py")
            for n in (
                "fit_stress",
                "reference_fit_candidate",
                "reference_fit",
                "reference_statistics",
                "repeated_exposure",
                "multiplane_reference",
            )
        ],
    ]
    report = {
        "status": "complete",
        "experiment": "fixed_candidate_model_and_noise_stress",
        "config": config,
        "solver_order": ["legacy", "candidate"],
        "conditions": conditions,
        "sha256": {str(p): digest(p) for p in paths},
        "production_solver_modified": False,
        "candidate_retuned": False,
        "training_performed": False,
        "correction_applied": False,
        "calibration_certified": False,
        "limitations": [
            "prespecified synthetic stress levels, not real-camera coverage",
            "noise estimator and thresholds intentionally retain original homoscedastic stationary assumptions",
            "no oracle covariance test asserted for misspecified noise models",
            "nominal fourth-plane alarms under reference mismatch are inconsistency signals, not ordinary false positives",
            "5mm translation scoring cutoff is an audit marker, not an operational certification threshold",
            "32 independent batches per condition, conditions and offsets paired; bootstrap intervals approximate",
            "no reliability gate added; usable optimization output is not model validation",
        ],
    }
    Path("docs/evaluations/fit-stress-v1.json").write_text(
        json.dumps(report, indent=1, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
