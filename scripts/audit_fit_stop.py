"""New independent validation of a fixed candidate stop with unchanged diagnostic thresholds."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

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
    draw,
    thresholds,
)
from scripts.audit_exposure_uncertainty import seed_plan

RULES = ("legacy_plugin", "candidate_plugin", "legacy_oracle", "candidate_oracle")


def aggregate(matrices, offsets, indices):
    a = np.asarray(matrices)  # batch, case, rule, offset
    rows = []
    for k, offset in enumerate(offsets):
        rules = {}
        for j, rule in enumerate(RULES):
            x = a[:, :, j, k]
            alarms, known = (x == 1).sum(1), (x != -1).sum(1)
            denominator = known[indices].sum(1)
            sample = np.divide(
                alarms[indices].sum(1),
                denominator,
                out=np.full(denominator.shape, np.nan),
                where=denominator > 0,
            )
            valid = sample[np.isfinite(sample)]
            pooled = counts(x.ravel())
            pooled.pop("wilson_95_percent_converged")
            rules[rule] = {
                **pooled,
                "cluster_95_percent": np.quantile(valid, [0.025, 0.975]).tolist()
                if valid.size
                else None,
            }
        common = (a[:, :, 0, k] != -1) & (a[:, :, 1, k] != -1)
        recovered = (a[:, :, 0, k] == -1) & (a[:, :, 1, k] != -1)
        lost = (a[:, :, 0, k] != -1) & (a[:, :, 1, k] == -1)
        rows.append(
            {
                "offset_m": offset,
                "rules": rules,
                "common_usable": {
                    "cases": int(common.sum()),
                    "plugin_disagreements": int(np.sum(common & (a[:, :, 0, k] != a[:, :, 1, k]))),
                    "oracle_disagreements": int(np.sum(common & (a[:, :, 2, k] != a[:, :, 3, k]))),
                    "plugin_alarms": {
                        "legacy": int(np.sum(common & (a[:, :, 0, k] == 1))),
                        "candidate": int(np.sum(common & (a[:, :, 1, k] == 1))),
                    },
                },
                "newly_usable": {
                    "cases": int(recovered.sum()),
                    "candidate_plugin_alarms": int(np.sum(recovered & (a[:, :, 1, k] == 1))),
                    "candidate_oracle_alarms": int(np.sum(recovered & (a[:, :, 3, k] == 1))),
                },
                "lost_usable_cases": int(lost.sum()),
            }
        )
    return rows


def negative_controls(rays, planes, truth):
    tests = [
        ("single_plane", planes[:1], predict(rays, planes[:1], truth)),
        ("beyond_bias_limit", planes[:3], predict(rays, planes[:3], truth) + 0.2),
    ]
    return [
        {
            "name": name,
            "legacy": fit_reference(rays, ps, obs),
            "candidate": fit_reference_candidate(rays, ps, obs),
        }
        for name, ps, obs in tests
    ]


def main():
    config_path = Path("config/fit-stop-validation.json")
    config = json.loads(config_path.read_text())
    calibration, validation = seed_plan(config)
    indices = np.random.default_rng(config["cluster_bootstrap_seed"]).integers(
        0, config["batch_count"], (config["cluster_bootstrap_replicates"], config["batch_count"])
    )
    baseline_path = Path("docs/evaluations/redundant-reference-v1.json")
    baseline = json.loads(baseline_path.read_text())
    planes = [(np.array(p["normal"]), p["offset_m"]) for p in baseline["planes"]]
    truth = np.array(baseline["true_parameters"])
    offsets = config["fourth_plane_offsets_m"]
    conditions, controls = [], []
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
        controls.append({"patch": patch, "cases": negative_controls(rays, planes, truth)})
        for model in ("independent", "global_shared", "exposure_shared"):
            e = 0.001 if model == "exposure_shared" else 0.0
            g = 0.001 if model == "global_shared" else 0.0
            batches, matrices = [], []
            for batch_id, (seeds, panel) in enumerate(zip(calibration, validation)):
                observations = np.array([draw(clean, s, e, g) for s in seeds])
                estimated = estimate_repeated_noise(observations)
                plugin = [estimated[key] for key in SIGMA_KEYS]
                decisions = np.full((len(panel), len(RULES), len(offsets)), -1, dtype=int)
                cases = []
                for i, seed in enumerate(panel):
                    observed = draw(clean, seed, e, g)
                    fits = [
                        fit_reference(rays, planes[:3], observed[:3]),
                        fit_reference_candidate(rays, planes[:3], observed[:3]),
                    ]
                    case = {
                        "seed": seed,
                        "legacy": fits[0],
                        "candidate": fits[1],
                        "observations_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                        "solver_diagnostics": [],
                    }
                    predicted = []
                    for j, fit in enumerate(fits):
                        fourth = predict(rays, planes[3:], fit["parameters"])[0]
                        predicted.append(fourth)
                        residuals = (observed[3] + shifts - fourth).mean(1)
                        detail = {"residuals_m": residuals.tolist()}
                        usable = fit["status"] == "converged" if j == 0 else fit["status"] in USABLE
                        if usable:
                            unit = mean_prediction_uncertainty(
                                rays, planes, fit["parameters"], 1, 1, 1
                            )
                            coefs = [unit["variance_components_m2"][key] for key in VARIANCE_KEYS]
                            limits = [
                                float(thresholds(coefs, s, config["nominal_alpha"]))
                                for s in (plugin, [0.001, e, g])
                            ]
                            decisions[i, j] = (abs(residuals) > limits[0]).astype(int)
                            decisions[i, j + 2] = (abs(residuals) > limits[1]).astype(int)
                            detail.update({"variance_coefficients": coefs, "thresholds_m": limits})
                        detail["scoring_only_parameter_error"] = {
                            "translation_norm_m": float(
                                np.linalg.norm(np.array(fit["parameters"][:3]) - truth[:3])
                            ),
                            "rotation_vector_norm_rad": float(
                                np.linalg.norm(np.array(fit["parameters"][3:6]) - truth[3:6])
                            ),
                            "depth_bias_error_m": float(fit["parameters"][6] - truth[6]),
                        }
                        case["solver_diagnostics"].append(detail)
                    delta = np.array(fits[1]["parameters"]) - np.array(fits[0]["parameters"])
                    case.update(
                        {
                            "parameter_delta": delta.tolist(),
                            "max_heldout_prediction_delta_m": float(
                                np.max(abs(predicted[1] - predicted[0]))
                            ),
                            "decisions": decisions[i].tolist(),
                        }
                    )
                    cases.append(case)
                matrices.append(decisions)
                batches.append(
                    {
                        "batch_id": batch_id,
                        "estimate": estimated,
                        "calibration_seeds_inclusive": [seeds[0], seeds[-1]],
                        "validation_seeds_inclusive": [panel[0], panel[-1]],
                        "calibration_observations_sha256": hashlib.sha256(
                            observations.tobytes()
                        ).hexdigest(),
                        "cases": cases,
                    }
                )
                if (batch_id + 1) % 8 == 0:
                    print(patch, model, f"{batch_id + 1}/32 batches", flush=True)
            flat = [c for b in batches for c in b["cases"]]
            summary = aggregate(matrices, offsets, indices)
            conditions.append(
                {
                    "patch": patch,
                    "noise_model": model,
                    "batches": batches,
                    "summary": summary,
                    "fit_status_counts": {
                        name: dict(Counter(c[name]["status"] for c in flat))
                        for name in ("legacy", "candidate")
                    },
                    "parameter_delta_max_abs": np.max(
                        abs(np.array([c["parameter_delta"] for c in flat])), axis=0
                    ).tolist(),
                    "heldout_prediction_delta_m": distribution(
                        [c["max_heldout_prediction_delta_m"] for c in flat]
                    ),
                }
            )
            print(patch, model, conditions[-1]["fit_status_counts"], flush=True)
    paths = [
        Path(__file__),
        config_path,
        baseline_path,
        *[
            Path(f"scripts/{n}.py")
            for n in ("audit_exposure_batches", "audit_exposure_uncertainty")
        ],
        *[
            Path(f"src/embodied_agent/{n}.py")
            for n in (
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
        "experiment": "candidate_numerical_stop_new_batches",
        "config": config,
        "rule_order": list(RULES),
        "conditions": conditions,
        "negative_controls": controls,
        "sha256": {str(p): digest(p) for p in paths},
        "production_solver_modified": False,
        "historical_unknowns_reclassified": False,
        "training_performed": False,
        "correction_applied": False,
        "calibration_certified": False,
        "limitations": [
            "candidate numerical stationarity is not calibration certification",
            "Gaussian stationary synthetic model; no real-camera or model-mismatch robustness guarantee",
            "32 independent batches per condition; conditions paired; cluster intervals approximate",
            "overall conditional rates have changing usable denominators; common/newly usable subsets reported separately",
            "thresholds remain nominal plug-in 1%; no guarantee of true 1% false alarms",
            "two negative controls per patch do not exhaust boundary or degeneracy cases",
        ],
    }
    Path("docs/evaluations/fit-stop-validation-v1.json").write_text(
        json.dumps(report, indent=1, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
