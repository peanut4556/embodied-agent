"""Recover reserved capture sessions and consume recent synthetic evidence once."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from embodied_agent.capture_recovery import CaptureRecovery
from scripts.check_process_pair_capture import fast_factory, hung_factory
from scripts.check_reserved_two_stage import sources as parent_sources


def drive(manager, token):
    deadline = time.monotonic() + 4
    while manager.runtime.controller.status == "pending" and time.monotonic() < deadline:
        manager.step(token)
        time.sleep(0.0005)
    if manager.runtime.controller.status == "pending":
        manager.step(token, stop=True)


def run(mode, repetition):
    manager = CaptureRecovery(
        position_error_m=(0.001,) * 3, max_generations=2, capture_profile="reserved"
    )
    try:
        old = manager.start(hung_factory)
        assert old is not None
        drive(manager, old)
        old_backend, old_handoff = manager.backend, manager._handoff
        token = manager.start(fast_factory)
        assert token is not None and token != old
        assert old_backend.result()["workers_stopped"]
        assert old_handoff.state == "revoked"
        assert manager.runtime.controller.requests == []
        stale_attempt = manager.consume_confirmation(old)
        assert stale_attempt["reason"] == "session_mismatch"
        drive(manager, token)
        if mode == "delayed_receipt":
            time.sleep(0.03)
        receipt = manager.consume_confirmation(token)
        repeated = manager.consume_confirmation(token)
        confirmed = manager.runtime.controller.status == "confirmed_recent_evidence"
        if confirmed and mode == "delayed_receipt":
            assert receipt["reason"] == "expired"
        if receipt["status"] == "consumed":
            assert confirmed and not receipt["robot_action_authorized"]
            assert 0 < receipt["remaining_s"] <= 0.02
            assert repeated["reason"] == "already_consumed"
        else:
            assert "evidence" not in receipt
        assert manager.close()
        return {
            "mode": mode,
            "repetition": repetition,
            "old_token_attempt": stale_attempt,
            "receipt": receipt,
            "repeated_attempt": repeated,
            "manager": manager.result(),
        }
    finally:
        manager.close()


def sources():
    return parent_sources() | {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (
            "scripts/check_recovery_handoff.py",
            "src/embodied_agent/capture_recovery.py",
            "src/embodied_agent/recent_evidence_handoff.py",
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = sources()
    cases = [
        run(mode, repeat)
        for repeat in range(2)
        for mode in ("immediate_receipt", "delayed_receipt")
    ]
    assert sources() == before
    report = args.output / "report.json"
    report.write_text(json.dumps({"sources": before, "cases": cases}, indent=2) + "\n")
    summary = {
        "kind": "reserved_recovery_recent_evidence_handoff",
        "sources": before,
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "cases": [],
    }
    for case in cases:
        history = case["manager"]["history"]
        confirmation = history[-1]["result"]["confirmation"]
        summary["cases"].append(
            {
                "mode": case["mode"],
                "repetition": case["repetition"],
                "old_reason": history[0]["result"]["confirmation"]["reason"],
                "new_status": confirmation["status"],
                "new_reason": confirmation["reason"],
                "new_requests": len(confirmation["requests"]),
                "receipt_status": case["receipt"]["status"],
                "receipt_reason": case["receipt"]["reason"],
                "remaining_s": case["receipt"].get("remaining_s"),
                "repeated_reason": case["repeated_attempt"]["reason"],
                "old_token_reason": case["old_token_attempt"]["reason"],
                "all_workers_stopped": all(h["cleanup"]["workers_stopped"] for h in history),
            }
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cases"], indent=2))


if __name__ == "__main__":
    main()
