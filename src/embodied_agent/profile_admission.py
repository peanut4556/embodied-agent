"""Exclude empirically contradicted acquisition profiles before execution."""

import copy
import hashlib
import json
import math
from dataclasses import dataclass

from .cross_view import CrossViewConfirmation
from .reference_confirmation import point3
from .uncertainty_view import UncertaintyViewConfirmation


@dataclass(frozen=True)
class ProfileAdmission:
    profiles: tuple
    audit_json: str

    def result(self):
        return json.loads(self.audit_json)


def admit_profiles(budgets, calibration_ids, cases, *, min_cases=8, min_coverage=2 / 3):
    """Audit raw validation observations, including valid halves of missing pairs.

    A complete case has both views present at every supplied frame. Repeated
    frames do not increase the case count. Any measured bound violation excludes
    a profile. This is finite-set admission, not a statistical certification.
    """
    if not isinstance(budgets, dict) or not budgets:
        raise ValueError("named profile budgets required")
    if type(min_cases) is not int or min_cases < 1:
        raise ValueError("positive minimum case count required")
    if not math.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    bounds = {}
    for key, value in budgets.items():
        if not isinstance(key, str) or not key or value is None:
            raise ValueError("named explicit budgets required")
        bounds[key] = CrossViewConfirmation(0, position_error_m=value).position_error_m
    calibration_ids, cases = list(calibration_ids), list(cases)
    if not calibration_ids or any(not isinstance(x, str) or not x for x in calibration_ids):
        raise ValueError("calibration case identities required")
    if len(set(calibration_ids)) != len(calibration_ids):
        raise ValueError("duplicate calibration identity")
    seen = set(calibration_ids)
    audits = {
        key: {
            "complete_cases": 0,
            "missing_views": 0,
            "exceeded_views": 0,
            "max_error_m": [0.0] * 3,
        }
        for key in bounds
    }
    for case in cases:
        identity = case["id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("validation identities must be unique and disjoint from calibration")
        seen.add(identity)
        if not isinstance(case["samples"], list) or not case["samples"]:
            raise ValueError("nonempty case observations required")
        complete = dict.fromkeys(bounds, True)
        for row in case["samples"]:
            truth = point3(row["truth_xyz"])
            if set(row["profiles"]) != set(bounds):
                raise ValueError("every profile must be represented in each observation")
            for key, bound in bounds.items():
                for side in ("primary", "secondary"):
                    value = row["profiles"][key][side]
                    if value is None:
                        audits[key]["missing_views"] += 1
                        complete[key] = False
                        continue
                    point = point3(value)
                    error = [abs(p - t) for p, t in zip(point, truth)]
                    if not all(math.isfinite(e) for e in error):
                        raise ValueError("nonfinite error")
                    audits[key]["max_error_m"] = [
                        max(a, b) for a, b in zip(audits[key]["max_error_m"], error)
                    ]
                    audits[key]["exceeded_views"] += int(
                        any(e > b + 1e-12 for e, b in zip(error, bound))
                    )
        for key in bounds:
            audits[key]["complete_cases"] += int(complete[key])
    eligible = []
    for key, audit in audits.items():
        coverage = audit["complete_cases"] / len(cases) if cases else 0
        reasons = []
        if audit["complete_cases"] < min_cases or coverage < min_coverage:
            reasons.append("insufficient_complete_case_coverage")
        if audit["exceeded_views"]:
            reasons.append("observed_error_exceeds_frozen_budget")
        audit.update(coverage=coverage, eligible=not reasons, reasons=reasons)
        if not reasons:
            eligible.append((key, bounds[key]))
    evidence = json.dumps(
        {"budgets": bounds, "calibration_ids": calibration_ids, "cases": cases},
        sort_keys=True,
        allow_nan=False,
    )
    result = {
        "schema_version": 1,
        "certified": False,
        "case_count": len(cases),
        "min_cases": min_cases,
        "min_coverage": min_coverage,
        "evidence_sha256": hashlib.sha256(evidence.encode()).hexdigest(),
        "audits": audits,
        "eligible_profiles": dict(eligible),
    }
    return ProfileAdmission(tuple(eligible), json.dumps(result, allow_nan=False))


class QualifiedViewConfirmation:
    """Only admitted profiles reach acquisition; an empty set never acquires."""

    def __init__(self, start_time, admission, *, initial="coarse"):
        if not isinstance(admission, ProfileAdmission):
            raise TypeError("ProfileAdmission required")
        # Validate the clock even when no profile is available.
        CrossViewConfirmation(start_time)
        self.admission = admission
        profiles = dict(admission.profiles)
        self.check = None
        if profiles:
            selected = (
                initial
                if initial in profiles
                else min(profiles, key=lambda k: (math.hypot(*profiles[k]), k))
            )
            self.check = UncertaintyViewConfirmation(
                start_time, profiles, initial=selected, switch_on_missing=True
            )

    @property
    def status(self):
        return self.check.status if self.check is not None else "unconfirmed"

    def step(self, timestamp, locate, *, stop=False):
        if self.check is None:
            return self.status
        return self.check.step(timestamp, locate, stop=stop)

    def result(self):
        return copy.deepcopy(
            {
                "status": self.status,
                "admission": self.admission.result(),
                "execution": self.check.result() if self.check is not None else None,
                "reason": "no_qualified_profiles"
                if self.check is None
                else self.check.confirmation.reason,
            }
        )
