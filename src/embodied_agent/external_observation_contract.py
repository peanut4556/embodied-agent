"""Strict, offline-only external observation import. Integrity is not evidence validity."""

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

ROLES = ("noise", "geometry", "synchronization", "motion", "drift", "independence")
IDENTITY = (
    "data_kind",
    "sensor_id",
    "camera_id",
    "frame_id",
    "clock_id",
    "mount_id",
    "measurement_point",
    "position_semantics",
    "reference_origin_id",
)
NUMBERS = (
    "max_age_s",
    "max_skew_s",
    "sigma_m",
    "static_error_bound_m",
    "mount_error_bound_m",
    "frame_error_bound_m",
    "clock_error_bound_s",
    "speed_bound_m_s",
    "drift_rate_bound_m_s",
)
POLICY_FIELDS = {
    "schema_version",
    *IDENTITY,
    *NUMBERS,
    "noise_model",
    "nominal_camera_origin_m",
    "valid_from",
    "valid_until",
    "reference_time",
    "evidence",
}
OBS_FIELDS = {*IDENTITY, "observation_id", "measurement_time", "camera_time", "units", "position"}


class ContractError(ValueError):
    pass


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ContractError(f"nonfinite JSON constant: {value}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    except ContractError:
        raise
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON: {exc.msg}") from exc
    except (ValueError, RecursionError) as exc:
        raise ContractError("JSON numeric or nesting limit exceeded") from exc


def fields(value, required, location):
    if not isinstance(value, dict):
        raise ContractError(f"{location}: object required")
    missing, extra = required - value.keys(), value.keys() - required
    if missing or extra:
        raise ContractError(f"{location}: missing={sorted(missing)}, unknown={sorted(extra)}")


def finite_numeric(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def number(value, location):
    if not finite_numeric(value) or value < 0:
        raise ContractError(f"{location}: finite nonnegative number required")
    return float(value)


def timestamp(value, location):
    if not isinstance(value, str):
        raise ContractError(f"{location}: timezone-aware ISO timestamp required")
    try:
        result = datetime.fromisoformat(value)
        if result.tzinfo is None:
            raise ValueError("missing timezone")
        return result.astimezone(UTC)
    except (ValueError, OverflowError) as exc:
        raise ContractError(f"{location}: timezone-aware ISO timestamp required") from exc


def version(value):
    if type(value) is not int or value != 1:
        raise ContractError("schema_version: only integer version 1 supported")


def validate_policy(policy, root, as_of):
    fields(policy, POLICY_FIELDS, "policy")
    version(policy["schema_version"])
    for key in IDENTITY:
        if not isinstance(policy[key], str) or not policy[key].strip() or len(policy[key]) > 128:
            raise ContractError(f"policy.{key}: nonempty bounded identifier required")
    if policy["data_kind"] not in ("synthetic", "measured"):
        raise ContractError("policy.data_kind: synthetic or measured required")
    if policy["measurement_point"] != "camera_origin":
        raise ContractError("only upstream-normalized camera_origin measurements supported")
    if policy["position_semantics"] != "translation_from_nominal_camera_origin":
        raise ContractError("absolute world positions require explicit upstream conversion")
    origin = policy["nominal_camera_origin_m"]
    if (
        not isinstance(origin, list)
        or len(origin) != 3
        or not all(finite_numeric(x) for x in origin)
    ):
        raise ContractError("nominal_camera_origin_m: three finite coordinates required")
    if policy["noise_model"] != "isotropic_gaussian":
        raise ContractError("only declared isotropic_gaussian noise supported")
    for key in NUMBERS:
        number(policy[key], f"policy.{key}")
    start = timestamp(policy["valid_from"], "policy.valid_from")
    end = timestamp(policy["valid_until"], "policy.valid_until")
    reference = timestamp(policy["reference_time"], "policy.reference_time")
    if not start <= reference <= as_of <= end:
        raise ContractError("policy validity/reference time does not cover evaluation time")
    fields(policy["evidence"], set(ROLES), "policy.evidence")
    root = Path(root).resolve(strict=True)
    evidence = {}
    for role in ROLES:
        item = policy["evidence"][role]
        fields(item, {"path", "sha256"}, f"evidence.{role}")
        if not isinstance(item["path"], str) or not item["path"]:
            raise ContractError(f"evidence.{role}: relative file path required")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"evidence.{role}: path outside evidence root")
        target = (root / relative).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():
            raise ContractError(f"evidence.{role}: path outside evidence root or not a file")
        with target.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        expected = item["sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or actual != expected:
            raise ContractError(f"evidence.{role}: SHA-256 mismatch")
        evidence[role] = {"path": item["path"], "sha256": actual}
    return start, end, reference, evidence


def normalize_observation(record, policy, start, end, reference, as_of):
    fields(record, OBS_FIELDS, "observation")
    oid = record["observation_id"]
    if not isinstance(oid, str) or not oid.strip() or len(oid) > 128:
        raise ContractError("observation_id: nonempty bounded identifier required")
    for key in IDENTITY:
        if type(record[key]) is not str or record[key] != policy[key]:
            raise ContractError(f"observation.{key}: does not match policy")
    if record["units"] not in ("m", "mm"):
        raise ContractError("observation.units: explicit m or mm required")
    position = record["position"]
    if (
        not isinstance(position, list)
        or len(position) != 3
        or any(not finite_numeric(x) for x in position)
    ):
        raise ContractError("observation.position: three finite xyz coordinates required")
    measured = timestamp(record["measurement_time"], "measurement_time")
    camera = timestamp(record["camera_time"], "camera_time")
    if not all(start <= t <= end and reference <= t <= as_of for t in (measured, camera)):
        raise ContractError(
            "observation time is future, outside validity, or before drift reference"
        )
    clock_error = policy["clock_error_bound_s"]
    for time in (measured, camera):
        if (
            min(
                (time - start).total_seconds(),
                (end - time).total_seconds(),
                (time - reference).total_seconds(),
                (as_of - time).total_seconds(),
            )
            < clock_error
        ):
            raise ContractError(
                "timestamp uncertainty crosses validity, drift reference or evaluation time"
            )
    age = (as_of - min(measured, camera)).total_seconds() + clock_error
    if age > policy["max_age_s"]:
        raise ContractError("observation too old")
    skew = abs((measured - camera).total_seconds()) + policy["clock_error_bound_s"]
    if skew > policy["max_skew_s"]:
        raise ContractError("timestamp separation plus clock error exceeds synchronization budget")
    drift_age = (max(measured, camera) - reference).total_seconds() + clock_error
    components = {
        "static_m": float(policy["static_error_bound_m"]),
        "mount_m": float(policy["mount_error_bound_m"]),
        "frame_m": float(policy["frame_error_bound_m"]),
        "motion_m": policy["speed_bound_m_s"] * skew,
        "drift_m": policy["drift_rate_bound_m_s"] * drift_age,
    }
    bound = sum(components.values())
    if not math.isfinite(bound):
        raise ContractError("computed systematic budget is nonfinite")
    scale = 1.0 if record["units"] == "m" else 0.001
    return {
        "observation_id": oid,
        **{k: record[k] for k in IDENTITY},
        "measurement_time": measured.isoformat(),
        "camera_time": camera.isoformat(),
        "translation_m": [float(x) * scale for x in position],
        "nominal_camera_origin_m": [float(x) for x in policy["nominal_camera_origin_m"]],
        "axes": [0, 1, 2],
        "sigma_m": float(policy["sigma_m"]),
        "systematic_error_bound_m": bound,
        "budget_components_m": components,
        "effective_skew_s": skew,
        "age_s": age,
        "drift_age_s": drift_age,
    }


def import_observations(document, policy, evidence_root, as_of):
    """All-or-nothing batch import; never run the admission gate or apply corrections."""
    report = {
        "schema_version": 1,
        "status": "rejected",
        "normalized": [],
        "errors": [],
        "eligible_for_offline_check": False,
        "calibration_certified": False,
        "correction_applied": False,
        "evidence_contents_authenticated": False,
    }
    try:
        now = timestamp(as_of, "as_of")
        start, end, reference, evidence = validate_policy(policy, evidence_root, now)
        fields(document, {"schema_version", "observations"}, "document")
        version(document["schema_version"])
        observations = document["observations"]
        if not isinstance(observations, list) or not 1 <= len(observations) <= 10000:
            raise ContractError("observations: nonempty list with at most 10000 records required")
        normalized, seen = [], set()
        for index, record in enumerate(observations):
            try:
                item = normalize_observation(record, policy, start, end, reference, now)
                if item["observation_id"] in seen:
                    raise ContractError("duplicate observation_id in batch")
                seen.add(item["observation_id"])
                normalized.append(item)
            except ContractError as exc:
                report["errors"].append({"index": index, "reason": str(exc)})
        if not report["errors"]:
            report.update(
                status="validated",
                normalized=normalized,
                eligible_for_offline_check=True,
                evidence=evidence,
                as_of=now.isoformat(),
            )
    except (ContractError, OSError, RuntimeError) as exc:
        report["errors"].append({"index": None, "reason": str(exc)})
    return report
