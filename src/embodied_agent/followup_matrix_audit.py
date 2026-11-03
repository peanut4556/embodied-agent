"""Frozen membership contract for the version 1 follow-up stress matrix.

Keep this independent of the evaluator: changing its loops must not silently
change the set of scenarios required for a complete audit.
"""

from itertools import product

PROFILE = "deterministic_session_fault_matrix_v1"
FIELDS = ("delay", "pattern", "drops", "fault")
AXES = (
    (0.03125, 0.125, 0.28125),
    ("inside", "outside", "transient"),
    ("none", "every_third", "all"),
    ("none", "clock_jump", "crash", "stop"),
)


def audit_membership(rows):
    """Check each required tuple appears exactly once, independent of row order."""
    expected = tuple(product(*AXES))
    seen = {}
    invalid = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid.append({"index": index, "error": "scenario must be an object"})
            continue
        values = tuple(row.get(field) for field in FIELDS)
        errors = []
        for field, value, allowed in zip(FIELDS, values, AXES, strict=True):
            valid_type = type(value) in (int, float) if field == "delay" else type(value) is str
            if not valid_type or value not in allowed:
                errors.append(field)
        if errors:
            invalid.append(
                {"index": index, "error": "invalid scenario parameters", "fields": errors}
            )
            continue
        seen.setdefault(values, []).append(index)

    missing = [dict(zip(FIELDS, key, strict=True)) for key in expected if key not in seen]
    duplicates = [
        {"parameters": dict(zip(FIELDS, key, strict=True)), "indices": indices}
        for key, indices in seen.items()
        if len(indices) > 1
    ]
    return {
        "profile": PROFILE,
        "status": "rejected" if missing or duplicates or invalid else "verified",
        "expected_cases": len(expected),
        "observed_cases": len(rows),
        "unique_expected_cases": len(seen),
        "missing": missing,
        "duplicates": duplicates,
        "invalid": invalid,
    }
