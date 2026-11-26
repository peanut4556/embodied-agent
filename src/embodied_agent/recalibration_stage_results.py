"""Check recorded assertions of the fixed synthetic maintenance scenario."""

from .recovery_json import parse_recovery_json


def check_stage_results(stages):
    issues, results = [], {}
    for stage in stages:
        name = stage["name"]
        error = stage["expected_exit_code"] == 3
        try:
            payload = parse_recovery_json(stage["stderr"] if error else stage["stdout"])
            if not isinstance(payload, dict) or payload.get("ok") is not (not error):
                raise ValueError("invalid result envelope")
            if stage["stdout"] if error else stage["stderr"]:
                raise ValueError("unexpected output channel")
            if error:
                expected = "ValueError" if name == "preview-invalid" else "StaleRecalibration"
                if payload.get("cause_type") != expected:
                    raise ValueError("unexpected rejection reason")
            else:
                result = payload.get("result")
                if not isinstance(result, dict):
                    raise ValueError("missing stage result")
                results[name] = result
        except (ValueError, TypeError):
            issues.append({"stage": name, "code": "invalid_stage_payload"})

    def check(name, predicate):
        if name not in results:
            return
        try:
            valid = predicate(results[name])
        except (KeyError, TypeError, IndexError):
            valid = False
        if not valid:
            issues.append({"stage": name, "code": "scenario_assertion_mismatch"})

    def diagnostic(result, ready):
        return (
            result["ready_for_preview"] is ready
            and result["applied"] is False
            and type(result["eligible_count"]) is int
            and result["eligible_count"] == (2 if ready else 0)
            and [r["action"] for r in result["results"]] == ["pick", "verify"]
        )

    check(
        "diagnose-invalid",
        lambda r: (
            diagnostic(r, False)
            and "insufficient_calibration_samples" in r["results"][0]["reasons"]
            and "validation_exceeds_candidate" in r["results"][1]["reasons"]
        ),
    )
    check("diagnose-fixed", lambda r: diagnostic(r, True) and r["combined_error"] is None)
    check(
        "diagnose-reused-batches",
        lambda r: (
            diagnostic(r, False)
            and all("historical_batch_conflict" in row["reasons"] for row in r["results"])
        ),
    )

    def preview(result, degraded):
        return (
            result["applied"] is False
            and type(result["before_revision"]) is int
            and result["before_revision"] == 0
            and type(result["after_revision"]) is int
            and result["after_revision"] == 2
            and result["after_degraded"] == degraded
            and [c["action"] for c in result["changes"]] == ["pick", "verify"]
        )

    check("preview-fixed", lambda r: preview(r, ["place"]))
    check("preview-current", lambda r: preview(r, ["locate", "place"]))
    check(
        "apply-current",
        lambda r: (
            r["applied"] is True
            and "preview-current" in results
            and {k: v for k, v in r.items() if k != "applied"}
            == {k: v for k, v in results["preview-current"].items() if k != "applied"}
        ),
    )

    def status(result, drifted):
        rows = result["results"]
        expected = ["applied_then_degraded" if drifted else "applied_current", "applied_current"]
        return (
            result["status"] == "all_matched"
            and type(result["current_revision"]) is int
            and result["current_revision"] == 2
            and result["matched_count"] == 2
            and result["request_count"] == 2
            and result["atomic_application_verified"] is False
            and result["currently_blocked_actions"] == (["pick"] if drifted else [])
            and [r["action"] for r in rows] == ["pick", "verify"]
            and [r["status"] for r in rows] == expected
            and [r["matched_revision"] for r in rows] == [1, 2]
            and [r["currently_blocked"] for r in rows] == [drifted, False]
            and result["counts"]
            == {
                "applied_current": 1 if drifted else 2,
                "applied_then_degraded": 1 if drifted else 0,
                "superseded": 0,
                "batch_conflict": 0,
                "not_found": 0,
            }
        )

    check("status-applied", lambda r: status(r, False))
    check("status-after-drift", lambda r: status(r, True))
    if "preview-fixed" in results and "preview-current" in results:
        first, current = results["preview-fixed"], results["preview-current"]
        if first.get("checkpoint_sha256") == current.get("checkpoint_sha256") or first.get(
            "plan_sha256"
        ) != current.get("plan_sha256"):
            issues.append({"stage": "preview-current", "code": "preview_transition_mismatch"})
    return issues
