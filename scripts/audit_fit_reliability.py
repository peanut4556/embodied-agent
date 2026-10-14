"""Fresh-batch evaluation of frozen, truth-blind uncertainty rejection gates."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from embodied_agent.exposure_uncertainty import bound_repeated_noise
from embodied_agent.fit_reliability import translation_covariance, translation_gate
from embodied_agent.fit_stress import stress_geometry, stress_observation
from embodied_agent.memory_policy import digest
from embodied_agent.reference_fit import predict
from embodied_agent.reference_fit_candidate import USABLE, fit_reference_candidate
from embodied_agent.reference_statistics import mean_prediction_uncertainty
from scripts.audit_exposure_batches import SIGMA_KEYS, VARIANCE_KEYS, thresholds
from scripts.audit_exposure_uncertainty import seed_plan


def summarize(batches, indices, limit):
    result = {}
    for rule in ("residual_only", "plugin", "bounded"):
        rows = []
        for batch in batches:
            cases = batch["cases"]
            accepted = [c for c in cases if c["accepted"][rule]]
            safe = [c for c in cases if c["translation_error_m"] <= limit]
            rows.append(
                [
                    len(cases),
                    len(accepted),
                    sum(c["translation_error_m"] > limit for c in accepted),
                    len(safe),
                    sum(not c["accepted"][rule] for c in safe),
                    sum(c["accepted"]["residual_only"] and not c["accepted"][rule] for c in safe),
                ]
            )
        a = np.array(rows)
        totals = a.sum(0)
        sampled = a[indices].sum(1)
        intervals = {}
        for name, num, den in [
            ("acceptance", 1, 0),
            ("risk_among_accepted", 2, 1),
            ("safe_rejection", 4, 3),
        ]:
            rates = np.divide(
                sampled[:, num],
                sampled[:, den],
                out=np.full(len(sampled), np.nan),
                where=sampled[:, den] > 0,
            )
            valid = rates[np.isfinite(rates)]
            intervals[name] = np.quantile(valid, [0.025, 0.975]).tolist() if valid.size else None
        result[rule] = dict(
            zip(
                (
                    "cases",
                    "accepted",
                    "accepted_over_limit",
                    "within_limit",
                    "within_limit_rejected",
                    "within_limit_extra_rejections",
                ),
                totals.tolist(),
            )
        )
        result[rule]["cluster_95_percent"] = intervals
    return result


def main():
    config_path = Path("config/fit-reliability.json")
    config = json.loads(config_path.read_text())
    previous_path = Path("docs/evaluations/fit-stress-v1.json")
    previous = json.loads(previous_path.read_text())
    for path, h in previous["sha256"].items():
        if digest(path) != h:
            raise ValueError(f"previous implementation changed: {path}")
    baseline_path = Path("docs/evaluations/redundant-reference-v1.json")
    baseline = json.loads(baseline_path.read_text())
    planes = [(np.array(p["normal"]), p["offset_m"]) for p in baseline["planes"]]
    truth = np.array(baseline["true_parameters"])
    calibration, validation = seed_plan(config)
    indices = np.random.default_rng(config["cluster_bootstrap_seed"]).integers(
        0, config["batch_count"], (config["cluster_bootstrap_replicates"], config["batch_count"])
    )
    designs = [
        (patch, scenario, 1.0) for patch in ("wide", "narrow") for scenario in config["scenarios"]
    ]
    designs += [
        ("wide", scenario, config["quiet_noise_multiplier"])
        for scenario in ("baseline", "coherent_reference_shift")
    ]
    conditions = []
    for patch, scenario, multiplier in designs:
        noise_config = {
            **config,
            "pixel_sigma_m": config["pixel_sigma_m"] * multiplier,
            "exposure_sigma_m": config["exposure_sigma_m"] * multiplier,
        }
        ex, ey = (0.25, 0.2) if patch == "wide" else (0.025, 0.02)
        rays = np.array(
            [[x, y, -1.0] for x in np.linspace(-ex, ex, 21) for y in np.linspace(-ey, ey, 21)]
        )
        rays, actual = stress_geometry(rays, planes, scenario, config)
        clean = predict(rays, actual, truth)
        batches = []
        for bi, (seeds, panel) in enumerate(zip(calibration, validation)):
            repeated = np.array(
                [
                    stress_observation(clean, rays, s, scenario, "calibration", noise_config)
                    for s in seeds
                ]
            )
            bound = bound_repeated_noise(repeated, config["calibration_failure_probability"])
            plugin = [bound["estimate"][k] for k in SIGMA_KEYS]
            cases = []
            for seed in panel:
                observed = stress_observation(
                    clean, rays, seed, scenario, "validation", noise_config
                )
                fit = fit_reference_candidate(rays, planes[:3], observed[:3])
                p = np.array(fit["parameters"])
                usable = fit["status"] in USABLE
                case = {
                    "seed": seed,
                    "observations_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                    "fit": fit,
                    "translation_error_m": float(np.linalg.norm(p[:3] - truth[:3])),
                    "accepted": dict.fromkeys(("residual_only", "plugin", "bounded"), False),
                    "gates": {},
                }
                if usable:
                    residual = float(np.mean(observed[3] - predict(rays, planes[3:], p)[0]))
                    unit = mean_prediction_uncertainty(rays, planes, p, 1, 1, 1)
                    coefficients = [unit["variance_components_m2"][k] for k in VARIANCE_KEYS]
                    threshold = float(thresholds(coefficients, plugin, config["nominal_alpha"]))
                    clear = bool(abs(residual) <= threshold)
                    case.update(
                        residual_m=residual,
                        variance_coefficients=coefficients,
                        threshold_m=threshold,
                    )
                    case["accepted"]["residual_only"] = clear
                    for rule, sigmas in (("plugin", plugin), ("bounded", bound["upper_sigmas_m"])):
                        cov = translation_covariance(rays, planes[:3], p, sigmas)
                        gate = translation_gate(
                            cov,
                            config["scoring_translation_error_limit_m"],
                            config["parameter_failure_probability"],
                        )
                        case["gates"][rule] = {**gate, "covariance_m2": cov.tolist()}
                        case["accepted"][rule] = bool(clear and gate["passes"])
                cases.append(case)
            batches.append(
                {
                    "batch_id": bi,
                    "calibration_seeds_inclusive": [seeds[0], seeds[-1]],
                    "validation_seeds_inclusive": [panel[0], panel[-1]],
                    "calibration_observations_sha256": hashlib.sha256(
                        repeated.tobytes()
                    ).hexdigest(),
                    "noise_bound": bound,
                    "cases": cases,
                }
            )
        cases = [c for b in batches for c in b["cases"]]
        coverage = {
            rule: {
                "usable": sum(rule in c["gates"] for c in cases),
                "error_exceeds_radius": sum(
                    c["translation_error_m"] > c["gates"][rule]["radius_m"]
                    for c in cases
                    if rule in c["gates"]
                ),
            }
            for rule in ("plugin", "bounded")
        }
        condition = {
            "patch": patch,
            "scenario": scenario,
            "noise_multiplier": multiplier,
            "batches": batches,
            "summary": summarize(batches, indices, config["scoring_translation_error_limit_m"]),
            "coverage": coverage,
            "fit_status_counts": dict(Counter(c["fit"]["status"] for c in cases)),
        }
        conditions.append(condition)
        print(patch, scenario, multiplier, condition["summary"]["bounded"], flush=True)
    paths = [
        config_path,
        previous_path,
        baseline_path,
        Path(__file__),
        *[
            Path("src/embodied_agent") / (name + ".py")
            for name in (
                "fit_reliability",
                "fit_stress",
                "reference_fit",
                "reference_fit_candidate",
                "reference_statistics",
                "exposure_uncertainty",
                "repeated_exposure",
                "multiplane_reference",
            )
        ],
        *[
            Path("scripts") / (name + ".py")
            for name in ("audit_exposure_batches", "audit_exposure_uncertainty")
        ],
    ]
    report = {
        "status": "complete",
        "config": config,
        "conditions": conditions,
        "sha256": {str(p): digest(p) for p in paths},
        "training_performed": False,
        "correction_applied": False,
        "calibration_certified": False,
        "limitations": [
            "local Gaussian covariance with estimated linearization; not a nonlinear coverage guarantee",
            "variance upper bounds require stationary homoscedastic Gaussian assumptions violated by stress cases",
            "shared reference bias is unobservable without an independent external reference",
            "truth is used only for scoring, never admission; 5mm is an experimental tolerance",
            "conditions paired, 32 independent batches per condition; cluster intervals approximate",
            "only zero additional fourth-plane offset tested; previous detection experiment not repeated",
        ],
    }
    Path("docs/evaluations/fit-reliability-v1.json").write_text(
        json.dumps(report, indent=1, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
