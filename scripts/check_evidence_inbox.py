"""Real-process synthetic confirmation followed by delayed/revoked consumer dispatch."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.evidence_inbox import EvidenceInbox
from scripts.check_process_pair_capture import fast_factory
from scripts.check_recovery_handoff import drive
from scripts.check_recovery_handoff import sources as parent_sources


def run(mode, repetition):
    manager = CaptureRecovery(position_error_m=(0.001,) * 3, capture_profile="reserved")
    inbox = EvidenceInbox()
    calls = []

    def consume(receipt):
        calls.append(receipt["session_id"])
        if mode == "consumer_exception":
            raise RuntimeError("synthetic exception after recording evidence")

    try:
        token = manager.start(fast_factory)
        assert token is not None
        drive(manager, token)
        offered = manager.queue_confirmation(token, inbox)
        if offered["status"] == "queued":
            if mode == "delayed":
                time.sleep(0.03)
            elif mode == "new_session":
                new = manager.start(fast_factory)
                assert new is not None and new != token
                manager.step(new, stop=True)
        dispatched = inbox.dispatch(consume)
        repeated = inbox.dispatch(consume)
        assert len(calls) <= 1 and repeated["reason"] == "no_pending_evidence"
        if offered["status"] == "queued" and mode in ("delayed", "new_session"):
            assert not calls
            assert dispatched["reason"] == (
                "expired_in_queue" if mode == "delayed" else "session_revoked"
            )
        if mode == "consumer_exception" and calls:
            assert dispatched["status"] == "outcome_unknown"
        assert manager.close()
        return {
            "mode": mode,
            "repetition": repetition,
            "offered": offered,
            "dispatched": dispatched,
            "repeated": repeated,
            "callback_calls": len(calls),
            "inbox": inbox.result(),
            "manager": manager.result(),
        }
    finally:
        inbox.close()
        manager.close()


def sources():
    return parent_sources() | {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in ("scripts/check_evidence_inbox.py", "src/embodied_agent/evidence_inbox.py")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = sources()
    cases = [
        run(mode, repetition)
        for repetition in range(2)
        for mode in ("immediate", "delayed", "new_session", "consumer_exception")
    ]
    assert sources() == before
    report = args.output / "report.json"
    report.write_text(json.dumps({"sources": before, "cases": cases}, indent=2) + "\n")
    summary = {
        "kind": "in_process_evidence_consumer_boundary",
        "sources": before,
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "cases": [],
    }
    for case in cases:
        history = case["manager"]["history"]
        confirmation = history[0]["result"]["confirmation"]
        summary["cases"].append(
            {
                "mode": case["mode"],
                "repetition": case["repetition"],
                "confirmation_status": confirmation["status"],
                "confirmation_reason": confirmation["reason"],
                "requests": len(confirmation["requests"]),
                "offer_status": case["offered"]["status"],
                "offer_reason": case["offered"]["reason"],
                "dispatch_status": case["dispatched"]["status"],
                "dispatch_reason": case["dispatched"]["reason"],
                "callback_calls": case["callback_calls"],
                "repeat_reason": case["repeated"]["reason"],
                "all_workers_stopped": all(h["cleanup"]["workers_stopped"] for h in history),
            }
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cases"], indent=2))


if __name__ == "__main__":
    main()
