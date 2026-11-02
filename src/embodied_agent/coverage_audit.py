"""Independent offline coverage interval reconstruction from outcome records."""

import math
from collections import Counter
from itertools import pairwise


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), "finite number required")
    return value


def audit_coverage(metrics, *, status=None):
    """Recompute coverage, not frame validity or authenticity. Never trust stored sums."""
    _require(type(metrics) is dict, "coverage must be an object")
    if "evidence" not in metrics:
        return {"status": "unavailable", "reason": "legacy_coverage_without_records"}
    try:
        return _audit(metrics, status)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f"malformed coverage evidence: {exc}") from exc


def _audit(metrics, status):
    _require(
        type(metrics["schema_version"]) is int and metrics["schema_version"] == 1,
        "unsupported coverage schema",
    )
    _require(metrics["definition"] == "consecutive_valid_exposure_intervals", "unknown definition")
    evidence = metrics["evidence"]
    _require(
        type(evidence["schema_version"]) is int and evidence["schema_version"] == 1,
        "unsupported evidence schema",
    )
    start, end = _number(evidence["start"]), _number(evidence["deadline"])
    _require(start < end, "invalid window")
    records = evidence["records"]
    _require(type(records) is list, "records must be a list")
    counts = Counter(dict.fromkeys(("valid", "missing", "rejected", "expired"), 0))
    valid = []
    for row in records:
        outcome = row["outcome"]
        _require(outcome in counts, "unknown record outcome")
        counts[outcome] += 1
        if outcome == "valid":
            exposure = _number(row["exposure"])
            _require(
                start < exposure < end and (not valid or exposure > valid[-1]),
                "invalid or unordered exposure",
            )
            valid.append(exposure)
        else:
            _require(row["exposure"] is None, "invalid record has an exposure")
    # Construct eligible edges from adjacent ledger entries, then compute their
    # union offline. Do not reuse the online accumulator or its intervals.
    edges = [
        (a["exposure"], b["exposure"])
        for a, b in pairwise(records)
        if a["outcome"] == b["outcome"] == "valid" and b["exposure"] - a["exposure"] <= 0.25
    ]
    union = []
    for left, right in sorted(edges):
        if union and left <= union[-1][1]:
            union[-1][1] = max(right, union[-1][1])
        else:
            union.append([left, right])
    covered = math.fsum(right - left for left, right in union)
    boundaries = [start] + [x for interval in union for x in interval] + [end]
    longest = max(boundaries[i + 1] - boundaries[i] for i in range(0, len(boundaries) - 1, 2))
    duration = end - start
    expected = {
        "planned_seconds": duration,
        "covered_seconds": covered,
        "uncovered_seconds": duration - covered,
        "coverage_fraction": covered / duration,
        "longest_uncovered_seconds": longest,
    }
    for key, value in expected.items():
        _require(
            math.isclose(_number(metrics[key]), value, rel_tol=0, abs_tol=1e-9),
            f"coverage mismatch: {key}",
        )
    _require(
        type(metrics["counts"]) is dict
        and set(metrics["counts"]) == set(counts)
        and all(
            type(metrics["counts"][key]) is int and metrics["counts"][key] == value
            for key, value in counts.items()
        ),
        "coverage counts mismatch",
    )
    _require(
        metrics["first_valid_exposure"] == (valid[0] if valid else None)
        and metrics["last_valid_exposure"] == (valid[-1] if valid else None),
        "first/last exposure mismatch",
    )
    completed = metrics["window_completed"]
    _require(type(completed) is bool, "invalid completion flag")
    if status is not None:
        _require(completed == (status == "completed"), "completion status mismatch")
    assessment = (
        "no_valid_observations"
        if not valid
        else "window_incomplete"
        if not completed
        else "observations_with_gaps"
        if covered < duration
        else "window_bracketed"
    )
    _require(metrics["assessment"] == assessment, "assessment mismatch")
    return {"status": "verified", "records_checked": len(records), "intervals": union, **expected}
