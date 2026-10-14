"""Independent synthetic external translation checks and explicit broken-assumption controls."""

import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_agent.exposure_uncertainty import bound_repeated_noise
from embodied_agent.external_reference import external_translation_gate
from embodied_agent.fit_reliability import translation_covariance, translation_gate
from embodied_agent.fit_stress import stress_observation
from embodied_agent.memory_policy import digest
from embodied_agent.reference_fit import predict
from embodied_agent.reference_fit_candidate import USABLE, fit_reference_candidate
from embodied_agent.reference_statistics import mean_prediction_uncertainty
from scripts.audit_exposure_batches import SIGMA_KEYS, VARIANCE_KEYS, thresholds
from scripts.audit_exposure_uncertainty import seed_plan

RULES = ("prior_gate", "independent_xyz", "shared_bias_xyz", "x_only_screen", "x_only_gate")


def summarize(batches, indices, limit):
    result = {}
    for rule in RULES:
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
                    sum(c["accepted"]["prior_gate"] and not c["accepted"][rule] for c in safe),
                ]
            )
        a = np.array(rows)
        total = a.sum(0)
        sampled = a[indices].sum(1)
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
                total.tolist(),
            )
        )
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
        result[rule]["cluster_95_percent"] = intervals
    return result


def main():
    cp = Path("config/external-reference.json")
    config = json.loads(cp.read_text())
    pp = Path("docs/evaluations/fit-reliability-v1.json")
    previous = json.loads(pp.read_text())
    for p, h in previous["sha256"].items():
        if digest(p) != h:
            raise ValueError(f"previous implementation changed: {p}")
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
    used = {s for group in (calibration, validation) for batch in group for s in batch}
    if used.intersection(anchor_seeds):
        raise ValueError("external observations must use separate random streams")
    indices = np.random.default_rng(config["cluster_bootstrap_seed"]).integers(
        0, config["batch_count"], (config["cluster_bootstrap_replicates"], config["batch_count"])
    )
    conditions = []
    for design in config["designs"]:
        shift = np.array(design["reference_shift_m"])
        actual = [(n, float(d + n @ shift)) for n, d in planes]
        clean = predict(rays, actual, truth)
        noise_config = {
            **config,
            "pixel_sigma_m": config["pixel_sigma_m"] * design["noise_multiplier"],
            "exposure_sigma_m": config["exposure_sigma_m"] * design["noise_multiplier"],
        }
        batches = []
        for bi, (seeds, panel) in enumerate(zip(calibration, validation)):
            repeated = np.array(
                [
                    stress_observation(clean, rays, s, "baseline", "calibration", noise_config)
                    for s in seeds
                ]
            )
            bound = bound_repeated_noise(repeated, config["calibration_failure_probability"])
            sigmas = [bound["estimate"][k] for k in SIGMA_KEYS]
            cases = []
            for i, seed in enumerate(panel):
                observed = stress_observation(
                    clean, rays, seed, "baseline", "validation", noise_config
                )
                fit = fit_reference_candidate(rays, planes[:3], observed[:3])
                p = np.array(fit["parameters"])
                anchor_seed = anchor_seeds[bi * len(panel) + i]
                # Truth appears only in synthetic observation generation and offline scoring.
                anchor = truth[:3] + np.random.default_rng(anchor_seed).normal(
                    0, config["anchor_sigma_m"], 3
                )
                shared = anchor - shift  # Deliberately broken independence/bias-budget control.
                external = {}
                for name, measurement, axes in [
                    ("independent_xyz", anchor, [0, 1, 2]),
                    ("shared_bias_xyz", shared, [0, 1, 2]),
                    ("x_only", anchor[:1], [0]),
                ]:
                    external[name] = external_translation_gate(
                        p[:3],
                        measurement,
                        axes,
                        config["anchor_sigma_m"],
                        config["scoring_translation_error_limit_m"],
                        config["anchor_failure_probability"],
                        config["anchor_systematic_error_bound_m"],
                    )
                prior = False
                diagnostic = {}
                if fit["status"] in USABLE:
                    residual = float(np.mean(observed[3] - predict(rays, planes[3:], p)[0]))
                    unit = mean_prediction_uncertainty(rays, planes, p, 1, 1, 1)
                    coefficients = [unit["variance_components_m2"][k] for k in VARIANCE_KEYS]
                    threshold = float(thresholds(coefficients, sigmas, config["nominal_alpha"]))
                    cov = translation_covariance(rays, planes[:3], p, bound["upper_sigmas_m"])
                    gate = translation_gate(
                        cov,
                        config["scoring_translation_error_limit_m"],
                        config["parameter_failure_probability"],
                    )
                    prior = bool(abs(residual) <= threshold and gate["passes"])
                    diagnostic = {
                        "residual_m": residual,
                        "variance_coefficients": coefficients,
                        "threshold_m": threshold,
                        "covariance_m2": cov.tolist(),
                        "gate": gate,
                    }
                accepted = {
                    "prior_gate": prior,
                    "independent_xyz": prior and external["independent_xyz"]["passes"],
                    "shared_bias_xyz": prior and external["shared_bias_xyz"]["passes"],
                    "x_only_screen": prior and external["x_only"]["projected_passes"],
                    "x_only_gate": prior and external["x_only"]["passes"],
                }
                cases.append(
                    {
                        "seed": seed,
                        "observations_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                        "anchor_seed": anchor_seed,
                        "independent_anchor_m": anchor.tolist(),
                        "shared_bias_anchor_m": shared.tolist(),
                        "fit": fit,
                        "prior_diagnostic": diagnostic,
                        "external_checks": external,
                        "accepted": accepted,
                        "translation_error_m": float(np.linalg.norm(p[:3] - truth[:3])),
                        "independent_anchor_error_m": float(np.linalg.norm(anchor - truth[:3])),
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
        condition = {
            "design": design,
            "batches": batches,
            "summary": summarize(batches, indices, config["scoring_translation_error_limit_m"]),
        }
        conditions.append(condition)
        print(
            design["name"],
            {k: (v["accepted"], v["accepted_over_limit"]) for k, v in condition["summary"].items()},
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
        "synthetic_external_sensor_only": True,
        "training_performed": False,
        "correction_applied": False,
        "calibration_certified": False,
        "limitations": [
            "known common frame/origin/extrinsics and isotropic Gaussian external noise assumed, not measured",
            "only translation observed; rotation and depth bias are not externally validated",
            "shared-bias control violates the declared zero bias budget; x-only screen is not a 3D admission rule",
            "external gate never consumes truth; truth only generates observations and scores outcomes",
            "confidence ball is a marginal noise bound, not a conditional error rate among selected cases",
            "paired conditions; 32 independent batches each; bootstrap intervals are descriptive, zero counts do not prove zero risk",
        ],
    }
    Path("docs/evaluations/external-reference-v1.json").write_text(
        json.dumps(report, indent=1, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
