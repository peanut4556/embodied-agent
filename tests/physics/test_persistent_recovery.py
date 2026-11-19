import copy
import unittest

from scripts.check_persistent_recovery import first_stage_verified, summarize


def fixture():
    def reservation(slot):
        return {"pool_id": "pool", "slot": slot, "allowance_ns": 5_000_000, "request_id": str(slot)}

    first = {
        "startup_failed": True,
        "timeout_observed": True,
        "closed": True,
        "pool": {"reserved_slots": 1},
        "manager": {
            "history": [{"cleanup": {"workers_stopped": True}}],
            "wait_cpu_budget": {"last_persistent_reservation": reservation(1)},
        },
    }
    trace = [
        {
            "reservation_wait": {
                "request_id": i,
                "wait_cpu_budget_s": 0.005,
                "effective_spin_window_s": 0.0015 if i <= 3 else 0,
                "wait_cpu_s": 0.002,
                "wait_cpu_spent_s": 0.002 * i,
            }
        }
        for i in range(1, 37)
    ]
    recovered = {
        "closed": True,
        "old_session_attempt": {"reason": "session_mismatch"},
        "confirmation": {"status": "confirmed_recent_evidence", "requests": list(range(36))},
        "receipt": {"status": "consumed"},
        "repeat": {"reason": "already_consumed"},
        "manager": {
            "history": [{"cleanup": {"workers_stopped": True, "timing_trace": trace}}],
            "wait_cpu_budget": {"last_persistent_reservation": reservation(2)},
        },
    }
    refused = {
        "refusal": "persistent wait budget exhausted",
        "backend_created": False,
        "pool": {"pool_id": "pool", "reserved_slots": 2, "remaining_ns": 0},
    }
    return first, recovered, refused


class PersistentRecoveryTests(unittest.TestCase):
    def test_complete_chain_and_missing_confirmation_are_distinct(self):
        first, second, third = fixture()
        self.assertTrue(summarize(first, second, third, [74, 0, 0])["full_recovery_passed"])
        second["confirmation"]["status"] = "rejected"
        second["receipt"]["status"] = "refused"
        result = summarize(first, second, third, [74, 0, 0])
        self.assertTrue(result["integrity_verified"])
        self.assertFalse(result["full_recovery_passed"])

    def test_cleanup_gate_blocks_restart(self):
        first, _, _ = fixture()
        self.assertFalse(first_stage_verified(first, 0))
        first["manager"]["history"][0]["cleanup"]["workers_stopped"] = False
        self.assertFalse(first_stage_verified(first, 74))

    def test_wrong_pool_reused_slot_or_spawn_on_refusal_rejected(self):
        for fault in ("pool", "slot", "spawn"):
            first, second, third = copy.deepcopy(fixture())
            if fault == "pool":
                third["pool"]["pool_id"] = "other"
            elif fault == "slot":
                second["manager"]["wait_cpu_budget"]["last_persistent_reservation"]["slot"] = 1
            else:
                third["backend_created"] = True
            self.assertFalse(summarize(first, second, third, [74, 0, 0])["integrity_verified"])

    def test_wrong_fallback_or_missing_telemetry_rejected(self):
        for fault in ("strategy", "missing"):
            first, second, third = fixture()
            trace = second["manager"]["history"][0]["cleanup"]["timing_trace"]
            if fault == "strategy":
                trace[-1]["reservation_wait"]["effective_spin_window_s"] = 0.0015
            else:
                trace.pop()
            self.assertFalse(summarize(first, second, third, [74, 0, 0])["integrity_verified"])


if __name__ == "__main__":
    unittest.main()
