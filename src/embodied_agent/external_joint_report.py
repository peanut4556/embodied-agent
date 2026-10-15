"""Pair raw visual data with revalidated external observations, for offline reporting only."""

import math

import numpy as np

from .exposure_uncertainty import bound_repeated_noise
from .external_observation_contract import (
    ContractError,
    fields,
    import_observations,
    timestamp,
    version,
)
from .external_reference import external_translation_gate
from .fit_reliability import translation_covariance, translation_gate
from .reference_fit import predict
from .reference_fit_candidate import USABLE, fit_reference_candidate
from .reference_statistics import mean_prediction_uncertainty

IDENTITY = ("data_kind", "camera_id", "frame_id", "clock_id", "reference_origin_id")
GATES = {
    "translation_limit_m": 0.005,
    "calibration_failure_probability": 0.005,
    "parameter_failure_probability": 0.045,
    "external_failure_probability": 0.01,
}


def numeric_array(value, shape, location):
    def numeric_tree(x):
        if isinstance(x, list):
            return all(numeric_tree(y) for y in x)
        return type(x) in (float, int)

    if not isinstance(value, list) or not numeric_tree(value):
        raise ContractError(f"{location}: numeric array required (no booleans/strings)")
    try:
        array = np.asarray(value, dtype=float)
    except (ValueError, OverflowError) as exc:
        raise ContractError(f"{location}: invalid numeric array") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractError(f"{location}: finite shape {shape} required")
    return array


def finite_payload(value):
    if isinstance(value, dict):
        return all(finite_payload(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_payload(v) for v in value)
    if isinstance(value, (float, int)):
        return math.isfinite(value)
    return True


def prepare_visual(document, imported, policy):
    fields(
        document, {"schema_version", "metadata", "rays", "planes", "calibration", "fits"}, "visual"
    )
    version(document["schema_version"])
    metadata = document["metadata"]
    fields(
        metadata, {*IDENTITY, "nominal_camera_origin_m", "position_semantics"}, "visual.metadata"
    )
    reference = imported[0]
    for key in (*IDENTITY, "position_semantics"):
        if type(metadata[key]) is not str or metadata[key] != reference[key]:
            raise ContractError(f"visual.metadata.{key}: external policy mismatch")
    origin = numeric_array(metadata["nominal_camera_origin_m"], (3,), "visual origin")
    if origin.tolist() != reference["nominal_camera_origin_m"] or origin.tolist() != [
        0.4,
        0.0,
        1.2,
    ]:
        raise ContractError(
            "visual origin must match external policy and fixed solver origin [0.4,0,1.2] m"
        )
    n = len(document["rays"]) if isinstance(document["rays"], list) else 0
    if not 7 <= n <= 1000:
        raise ContractError("visual.rays: 7..1000 rays required")
    rays = numeric_array(document["rays"], (n, 3), "visual.rays")
    if not np.all(rays[:, 2] == -1):
        raise ContractError("visual.rays: normalized z=-1 convention required")
    if not isinstance(document["planes"], list) or len(document["planes"]) != 4:
        raise ContractError("four nominal planes required")
    planes = []
    for item in document["planes"]:
        fields(item, {"normal", "offset_m"}, "plane")
        normal = numeric_array(item["normal"], (3,), "plane.normal")
        offset = numeric_array([item["offset_m"]], (1,), "plane.offset_m")[0]
        if not np.isclose(np.linalg.norm(normal), 1.0, rtol=0, atol=1e-10):
            raise ContractError("plane normal must have unit length")
        planes.append((normal, float(offset)))
    cal = document["calibration"]
    fields(cal, {"start_time", "end_time", "depths_m"}, "visual.calibration")
    start, end = (
        timestamp(cal["start_time"], "calibration.start"),
        timestamp(cal["end_time"], "calibration.end"),
    )
    valid_from = timestamp(policy["valid_from"], "policy.valid_from")
    valid_until = timestamp(policy["valid_until"], "policy.valid_until")
    if not valid_from <= start <= end <= valid_until:
        raise ContractError("visual calibration is outside the declared policy validity window")
    repeats = len(cal["depths_m"]) if isinstance(cal["depths_m"], list) else 0
    if not 32 <= repeats <= 1024 or start > end:
        raise ContractError("calibration requires 32..1024 repeats and ordered times")
    repeated = numeric_array(cal["depths_m"], (repeats, 4, n), "calibration.depths_m")
    if np.any(repeated <= 0):
        raise ContractError("calibration depths must be positive metres")
    fits = document["fits"]
    if not isinstance(fits, list) or len(fits) != len(imported):
        raise ContractError("exact one-to-one visual/external record coverage required")
    lookup = {o["observation_id"]: o for o in imported}
    paired = []
    seen_fits = set()
    seen_observations = set()
    for record in fits:
        fields(record, {"fit_id", "observation_id", "camera_time", "depths_m"}, "visual.fit")
        fit_id, oid = record["fit_id"], record["observation_id"]
        if (
            not isinstance(fit_id, str)
            or not fit_id.strip()
            or len(fit_id) > 128
            or fit_id in seen_fits
        ):
            raise ContractError("fit_id must be unique nonempty bounded string")
        if not isinstance(oid, str) or oid not in lookup or oid in seen_observations:
            raise ContractError("unmatched or reused observation_id")
        external = lookup[oid]
        camera = timestamp(record["camera_time"], "visual.camera_time")
        if camera != timestamp(external["camera_time"], "external.camera_time") or end >= camera:
            raise ContractError("camera time mismatch or calibration overlaps validation")
        depths = numeric_array(record["depths_m"], (4, n), "visual.depths_m")
        if np.any(depths <= 0):
            raise ContractError("visual depths must be positive metres")
        seen_fits.add(fit_id)
        seen_observations.add(oid)
        paired.append((fit_id, external, depths))
    return rays, planes, repeated, paired


def evaluate_pair(rays, planes, depths, external, bound, fit_id):
    result = {
        "fit_id": fit_id,
        "observation_id": external["observation_id"],
        "camera_time": external["camera_time"],
        "status": "unknown",
        "passes_offline_checks": False,
        "calibration_certified": False,
        "correction_applied": False,
    }
    try:
        fit = fit_reference_candidate(rays, planes[:3], depths[:3])
        if not finite_payload(fit):
            raise ValueError("nonfinite fit output")
        result["fit"] = fit
        if fit["status"] not in USABLE:
            result["reason"] = "visual solver not usable"
            return result
        p = np.array(fit["parameters"])
        estimate = bound["estimate"]
        residual = float(np.mean(depths[3] - predict(rays, planes[3:], p)[0]))
        uncertainty = mean_prediction_uncertainty(
            rays,
            planes,
            p,
            estimate["sigma_pixel_m"],
            estimate["sigma_exposure_m"],
            estimate["sigma_global_m"],
        )
        threshold = uncertainty["two_sided_99_percent_threshold_m"]
        covariance = translation_covariance(rays, planes[:3], p, bound["upper_sigmas_m"])
        precision = translation_gate(
            covariance, GATES["translation_limit_m"], GATES["parameter_failure_probability"]
        )
        external_check = external_translation_gate(
            p[:3],
            external["translation_m"],
            [0, 1, 2],
            external["sigma_m"],
            GATES["translation_limit_m"],
            GATES["external_failure_probability"],
            external["systematic_error_bound_m"],
        )
        if not all(
            math.isfinite(x)
            for x in (residual, threshold, precision["radius_m"], external_check["error_upper_m"])
        ):
            raise ValueError("nonfinite diagnostic")
        residual_pass = bool(abs(residual) <= threshold)
        passes = bool(residual_pass and precision["passes"] and external_check["passes"])
        result.update(
            status="passed" if passes else "rejected",
            passes_offline_checks=passes,
            residual_check={
                "residual_m": residual,
                "threshold_m": threshold,
                "passes": residual_pass,
            },
            precision_check={**precision, "covariance_m2": covariance.tolist()},
            external_check=external_check,
        )
    except (ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        result["reason"] = f"numerical evaluation unavailable: {exc}"
    return result


def build_joint_report(observations, policy, evidence_root, visual, as_of):
    imported = import_observations(observations, policy, evidence_root, as_of)
    report = {
        "schema_version": 1,
        "status": "rejected",
        "import": imported,
        "pairing_status": "not_evaluated",
        "results": [],
        "errors": [],
        "gate_configuration": GATES.copy(),
        "calibration_certified": False,
        "correction_applied": False,
        "evidence_contents_authenticated": False,
    }
    if imported["status"] != "validated":
        report["errors"] = [{"reason": "external import rejected"}]
        return report
    try:
        rays, planes, repeated, paired = prepare_visual(visual, imported["normalized"], policy)
        bound = bound_repeated_noise(repeated, GATES["calibration_failure_probability"])
        if not finite_payload(bound):
            raise ContractError("nonfinite noise calibration output")
    except (ContractError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        report.update(pairing_status="rejected", errors=[{"reason": str(exc)}])
        return report
    results = [
        evaluate_pair(rays, planes, depths, external, bound, fit_id)
        for fit_id, external, depths in paired
    ]
    report.update(
        status="evaluated",
        pairing_status="matched",
        noise_bound=bound,
        results=results,
        summary={
            status: sum(r["status"] == status for r in results)
            for status in ("passed", "rejected", "unknown")
        },
    )
    return report
