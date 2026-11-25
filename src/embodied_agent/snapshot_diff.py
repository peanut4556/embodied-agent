"""Offline comparison of verified audit snapshots; findings require human context."""

from .operation_snapshot import read_snapshot


def compare_snapshots(before, after, *, allow_source_override=False):
    if type(allow_source_override) is not bool:
        raise TypeError("allow_source_override must be boolean")
    first, second = read_snapshot(before), read_snapshot(after)
    a, b = first["snapshot"], second["snapshot"]
    source_flags = []
    left_id, right_id = a.get("source_id"), b.get("source_id")
    if left_id is None or right_id is None:
        relation = "unavailable"
        source_flags.append("source_identity_unavailable")
    elif left_id != right_id:
        relation = "mismatched_declared"
        source_flags.append("source_identity_mismatch")
    else:
        relation = "matched_declared"
    left_sequence, right_sequence = a.get("capture_sequence"), b.get("capture_sequence")
    order_relation = "unavailable"
    if relation == "matched_declared" and left_sequence is not None and right_sequence is not None:
        if right_sequence < left_sequence:
            order_relation = "reversed_declared"
            source_flags.append("capture_order_reversed")
        elif right_sequence == left_sequence and first["sha256"] != second["sha256"]:
            order_relation = "reused_declared"
            source_flags.append("capture_sequence_reused")
        elif right_sequence == left_sequence:
            order_relation = "same_capture"
        else:
            order_relation = "increasing_declared"
    source = {
        "before_id": left_id,
        "after_id": right_id,
        "relation": relation,
        "order_relation": order_relation,
        "before_sequence": left_sequence,
        "after_sequence": right_sequence,
        "flags": source_flags,
        "override": allow_source_override,
    }
    if source_flags and not allow_source_override:
        return {
            "before_sha256": first["sha256"],
            "after_sha256": second["sha256"],
            "same_source_verified": False,
            "physical_outcome_verified": False,
            "source": source,
            "comparison_performed": False,
            "needs_review": True,
            "counts": None,
            "changes": [],
            "budget_changes": {},
            "budget_flags": [],
        }
    old = {row["receipt_key"]: row for row in a["items"]}
    new = {row["receipt_key"]: row for row in b["items"]}
    changes, unchanged = [], 0
    for key in sorted(old.keys() | new.keys()):
        left, right = old.get(key), new.get(key)
        flags = []
        if left is None:
            kind = "added"
            if right["error"] is not None:
                flags.append("unreadable_source")
        elif right is None:
            kind = "removed"
            flags.append("receipt_missing")
        elif left["error"] is not None or right["error"] is not None:
            kind = "uncertain"
            flags.append("unreadable_source")
        else:
            previous, current = left["data"], right["data"]
            if previous == current:
                if left["source_sha256"] == right["source_sha256"]:
                    unchanged += 1
                    continue
                kind = "encoding_changed"
            else:
                kind = "changed"
                if previous["request_sha256"] != current["request_sha256"]:
                    flags.append("request_rebound")
                if previous["state"] in ("complete", "abandoned"):
                    flags.append("terminal_receipt_changed")
                if (
                    previous["state"] == "pending"
                    and current["state"] == "abandoned"
                    and current["resolution"]["previous_receipt_sha256"] != previous["sha256"]
                ):
                    flags.append("abandonment_predecessor_mismatch")
        changes.append(
            {
                "receipt_key": key,
                "kind": kind,
                "flags": flags,
                "before_state": None
                if left is None
                else ("invalid" if left["error"] else left["data"]["state"]),
                "after_state": None
                if right is None
                else ("invalid" if right["error"] else right["data"]["state"]),
            }
        )
    budget_flags = []
    budget_changes = {}
    if a["budget"]["error"] is not None or b["budget"]["error"] is not None:
        budget_flags.append("budget_unreadable")
    else:
        previous, current = a["budget"]["data"]["state"], b["budget"]["data"]["state"]
        budget_changes = {
            "before_revision": previous["revision"],
            "after_revision": current["revision"],
            "newly_degraded": sorted(set(current["degraded"]) - set(previous["degraded"])),
            "cleared_degraded": sorted(set(previous["degraded"]) - set(current["degraded"])),
        }
        if current["revision"] < previous["revision"]:
            budget_flags.append("budget_revision_decreased")
        history = previous["history"]
        if current["history"][: len(history)] != history:
            budget_flags.append("calibration_history_not_extended")
        if any(
            previous[k] != current[k]
            for k in ("fallback_s", "margin_s", "minimum_samples", "drift_policy")
        ):
            budget_flags.append("budget_configuration_changed")
        recalibrated = {row["action"] for row in current["history"][len(history) :]}
        if set(budget_changes["cleared_degraded"]) - recalibrated:
            budget_flags.append("degradation_cleared_without_new_calibration")
        changed_estimates = {
            name
            for name in previous["estimates"].keys() | current["estimates"].keys()
            if previous["estimates"].get(name) != current["estimates"].get(name)
        }
        if changed_estimates - recalibrated:
            budget_flags.append("estimate_changed_without_new_calibration")
    counts = {
        kind: sum(row["kind"] == kind for row in changes)
        for kind in ("added", "removed", "changed", "encoding_changed", "uncertain")
    }
    needs_review = bool(source_flags or budget_flags or any(row["flags"] for row in changes))
    return {
        "before_sha256": first["sha256"],
        "after_sha256": second["sha256"],
        "same_source_verified": False,
        "physical_outcome_verified": False,
        "source": source,
        "comparison_performed": True,
        "counts": {**counts, "unchanged": unchanged},
        "changes": changes,
        "budget_changes": budget_changes,
        "budget_flags": budget_flags,
        "needs_review": needs_review,
        "ignored_entries": {"before": a["ignored_entries"], "after": b["ignored_entries"]},
    }
