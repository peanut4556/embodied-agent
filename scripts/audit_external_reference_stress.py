"""Fresh visual batches, paired external-sensor stress, frozen admission checks."""

import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_agent.exposure_uncertainty import bound_repeated_noise
from embodied_agent.external_reference import external_translation_gate
from embodied_agent.external_reference_stress import external_observation
from embodied_agent.fit_reliability import translation_covariance, translation_gate
from embodied_agent.fit_stress import stress_observation
from embodied_agent.memory_policy import digest
from embodied_agent.reference_fit import predict
from embodied_agent.reference_fit_candidate import USABLE, fit_reference_candidate
from embodied_agent.reference_statistics import mean_prediction_uncertainty
from scripts.audit_exposure_batches import SIGMA_KEYS, VARIANCE_KEYS, thresholds
from scripts.audit_exposure_uncertainty import seed_plan


def summarize(batches, scenario, indices, limit):
    result = {}
    for rule in ("nominal", "budgeted"):
        rows = []
        for batch in batches:
            cases = batch["cases"]
            selected = [c for c in cases if c["external"][scenario][rule]["accepted"]]
            safe = [c for c in cases if c["translation_error_m"] <= limit]
            rows.append(
                [
                    len(cases),
                    len(selected),
                    sum(c["translation_error_m"] > limit for c in selected),
                    len(safe),
                    sum(not c["external"][scenario][rule]["accepted"] for c in safe),
                    sum(
                        c["prior_accepted"] and not c["external"][scenario][rule]["accepted"]
                        for c in safe
                    ),
                    sum(
                        c["external"][scenario]["anchor_error_m"]
                        > c["external"][scenario][rule]["sensor_radius_m"]
                        for c in cases
                    ),
                    sum(
                        c["translation_error_m"] > c["external"][scenario][rule]["error_upper_m"]
                        for c in cases
                    ),
                ]
            )
        a = np.array(rows)
        sampled = a[indices].sum(1)
        row = dict(
            zip(
                (
                    "cases",
                    "accepted",
                    "accepted_over_limit",
                    "within_limit",
                    "within_limit_rejected",
                    "within_limit_extra_rejections",
                    "anchor_outside_radius",
                    "translation_outside_upper",
                ),
                a.sum(0).tolist(),
            )
        )
        row["cluster_95_percent"] = {}
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
            row["cluster_95_percent"][name] = (
                np.quantile(valid, [0.025, 0.975]).tolist() if valid.size else None
            )
        result[rule] = row
    return result


def main():
    cp = Path("config/external-reference-stress.json")
    config = json.loads(cp.read_text())
    pp = Path("docs/evaluations/external-reference-v1.json")
    previous = json.loads(pp.read_text())
    for path, h in previous["sha256"].items():
        if digest(path) != h:
            raise ValueError(f"previous implementation changed: {path}")
    bp = Path("docs/evaluations/redundant-reference-v1.json")
    baseline = json.loads(bp.read_text())
    planes = [(np.array(p["normal"]), p["offset_m"]) for p in baseline["planes"]]
    truth = np.array(baseline["true_parameters"])
    rays = np.array(
        [[x, y, -1.0] for x in np.linspace(-0.25, 0.25, 21) for y in np.linspace(-0.2, 0.2, 21)]
    )
    calibration, validation = seed_plan(config)
    anchor_seeds = list(
        range(
            config["anchor_seed_start"],
            config["anchor_seed_start"] + config["batch_count"] * config["validation_per_batch"],
        )
    )
    if set(anchor_seeds) & {
        s for group in (calibration, validation) for batch in group for s in batch
    }:
        raise ValueError("overlapping anchor seeds")
    indices = np.random.default_rng(config["cluster_bootstrap_seed"]).integers(
        0, config["batch_count"], (config["cluster_bootstrap_replicates"], config["batch_count"])
    )
    conditions = []
    for design in config["designs"]:
        shift = np.array(design["reference_shift_m"])
        actual = [(n, float(d + n @ shift)) for n, d in planes]
        clean = predict(rays, actual, truth)
        nc = {
            **config,
            "pixel_sigma_m": config["pixel_sigma_m"] * design["noise_multiplier"],
            "exposure_sigma_m": config["exposure_sigma_m"] * design["noise_multiplier"],
        }
        batches = []
        for bi, (seeds, panel) in enumerate(zip(calibration, validation)):
            repeated = np.array(
                [stress_observation(clean, rays, s, "baseline", "calibration", nc) for s in seeds]
            )
            bound = bound_repeated_noise(repeated, config["calibration_failure_probability"])
            sigmas = [bound["estimate"][k] for k in SIGMA_KEYS]
            cases = []
            for i, seed in enumerate(panel):
                observed = stress_observation(clean, rays, seed, "baseline", "validation", nc)
                fit = fit_reference_candidate(rays, planes[:3], observed[:3])
                p = np.array(fit["parameters"])
                prior = False
                diagnostic = {}
                if fit["status"] in USABLE:
                    residual = float(np.mean(observed[3] - predict(rays, planes[3:], p)[0]))
                    unit = mean_prediction_uncertainty(rays, planes, p, 1, 1, 1)
                    coefs = [unit["variance_components_m2"][k] for k in VARIANCE_KEYS]
                    threshold = float(thresholds(coefs, sigmas, config["nominal_alpha"]))
                    cov = translation_covariance(rays, planes[:3], p, bound["upper_sigmas_m"])
                    gate = translation_gate(
                        cov,
                        config["scoring_translation_error_limit_m"],
                        config["parameter_failure_probability"],
                    )
                    prior = bool(abs(residual) <= threshold and gate["passes"])
                    diagnostic = {
                        "residual_m": residual,
                        "variance_coefficients": coefs,
                        "threshold_m": threshold,
                        "covariance_m2": cov.tolist(),
                        "gate": gate,
                    }
                anchor_seed = anchor_seeds[bi * len(panel) + i]
                external = {}
                for scenario in config["external_scenarios"]:
                    measured, bias = external_observation(
                        truth[:3], anchor_seed, i / (len(panel) - 1), scenario
                    )
                    row = {
                        "measurement_m": measured.tolist(),
                        "generated_bias_m": bias.tolist(),
                        "anchor_error_m": float(np.linalg.norm(measured - truth[:3])),
                    }
                    for rule, sigma, budget in [
                        ("nominal", config["anchor_sigma_m"], 0.0),
                        ("budgeted", scenario["budget_sigma_m"], scenario["budget_bias_m"]),
                    ]:
                        check = external_translation_gate(
                            p[:3],
                            measured,
                            [0, 1, 2],
                            sigma,
                            config["scoring_translation_error_limit_m"],
                            config["anchor_failure_probability"],
                            budget,
                        )
                        row[rule] = {**check, "accepted": bool(prior and check["passes"])}
                    external[scenario["name"]] = row
                cases.append(
                    {
                        "seed": seed,
                        "anchor_seed": anchor_seed,
                        "position_in_batch": i / (len(panel) - 1),
                        "observations_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                        "fit": fit,
                        "prior_diagnostic": diagnostic,
                        "prior_accepted": prior,
                        "external": external,
                        "translation_error_m": float(np.linalg.norm(p[:3] - truth[:3])),
                    }
                )
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
        summary = {
            s["name"]: summarize(
                batches, s["name"], indices, config["scoring_translation_error_limit_m"]
            )
            for s in config["external_scenarios"]
        }
        conditions.append({"design": design, "batches": batches, "summary": summary})
        print(
            design["name"],
            {
                k: {rule: (row["accepted"], row["accepted_over_limit"]) for rule, row in v.items()}
                for k, v in summary.items()
            },
            flush=True,
        )
    paths = [
        cp,
        pp,
        bp,
        Path(__file__),
        *[
            Path("src/embodied_agent") / (n + ".py")
            for n in (
                "external_reference_stress",
                "external_reference",
                "fit_reliability",
                "fit_stress",
                "exposure_uncertainty",
                "reference_fit",
                "reference_fit_candidate",
                "reference_statistics",
                "repeated_exposure",
                "multiplane_reference",
            )
        ],
        *[
            Path("scripts") / (n + ".py")
            for n in ("audit_exposure_batches", "audit_exposure_uncertainty")
        ],
    ]
    report = {
        "status": "complete",
        "config": config,
        "conditions": conditions,
        "sha256": {str(p): digest(p) for p in paths},
        "synthetic_only": True,
        "training_performed": False,
        "correction_applied": False,
        "calibration_certified": False,
        "limitations": [
            "scenario budgets fixed before evaluation, not inferred from observations or validated hardware specifications",
            "only x-directed reference shifts and low visual noise; rotation and depth bias not certified",
            "drift resets per independent batch; latency is a local constant-velocity observation model, not a dynamic camera simulation",
            "frame bias bound assumes world position norm at most 1.5m and known rotation-error bound",
            "paired mechanisms and shifts share observations; bootstrap resamples entire batches",
            "zero admissions means undefined conditional risk; zero observed failures does not prove zero population risk",
        ],
    }
    Path("docs/evaluations/external-reference-stress-v1.json").write_text(
        json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
