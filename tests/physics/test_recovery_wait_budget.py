import unittest
from unittest.mock import patch

from test_capture_recovery import bad_factory, factory

from embodied_agent.capture_recovery import CaptureRecovery


def manager(**kwargs):
    return CaptureRecovery(
        position_error_m=(0.001,) * 3,
        capture_profile="reserved",
        max_generations=2,
        wait_cpu_budget_s=0.01,
        **kwargs,
    )


class RecoveryBudgetTests(unittest.TestCase):
    def test_failed_start_reserves_share_and_second_gets_only_its_share(self):
        m = manager()
        try:
            self.assertIsNone(m.start(bad_factory))
            self.assertEqual(m.result()["wait_cpu_budget"]["unallocated_s"], 0.005)
            self.assertIsNotNone(m.start(factory))
            b = m.result()["wait_cpu_budget"]
            self.assertEqual([a["allowance_s"] for a in b["allocations"]], [0.005, 0.005])
            self.assertEqual(b["unallocated_s"], 0)
            self.assertTrue(m.trace_timing)
            b["allocations"].clear()
            self.assertEqual(len(m.result()["wait_cpu_budget"]["allocations"]), 2)
            m.close()
            with self.assertRaisesRegex(RuntimeError, "generation budget"):
                m.start(factory)
        finally:
            self.assertTrue(m.close())

    def test_cleanup_pending_does_not_allocate_or_spawn(self):
        m = manager()
        try:
            token = m.start(factory)
            m.step(token, stop=True)
            with (
                patch.object(
                    m.runtime, "close", return_value={"closed": True, "workers_stopped": False}
                ),
                patch("embodied_agent.capture_recovery.ProcessPairCapture") as spawn,
            ):
                self.assertIsNone(m.start(factory))
                spawn.assert_not_called()
            self.assertEqual(len(m.wait_budget_allocations), 1)
        finally:
            self.assertTrue(m.close())

    def test_spawn_exception_keeps_reservation_and_blocks_new_generation(self):
        m = manager()
        with (
            patch(
                "embodied_agent.capture_recovery.ProcessPairCapture",
                side_effect=RuntimeError("spawn"),
            ),
            self.assertRaises(RuntimeError),
        ):
            m.start(factory)
        self.assertEqual(m.result()["wait_cpu_budget"]["unallocated_s"], 0.005)
        with self.assertRaisesRegex(RuntimeError, "ownership unresolved"):
            m.start(factory)
        self.assertEqual(len(m.wait_budget_allocations), 1)

    def test_invalid_profile_budget_and_legacy_default(self):
        for value in (True, -1, float("nan"), 2):
            with self.assertRaises(ValueError):
                CaptureRecovery(
                    position_error_m=(0.001,) * 3,
                    capture_profile="reserved",
                    wait_cpu_budget_s=value,
                )
        with self.assertRaisesRegex(ValueError, "reserved"):
            CaptureRecovery(position_error_m=(0.001,) * 3, wait_cpu_budget_s=0)
        m = CaptureRecovery(position_error_m=(0.001,) * 3)
        self.assertIsNone(m.result()["wait_cpu_budget"]["total_threshold_s"])
        self.assertFalse(m.trace_timing)


if __name__ == "__main__":
    unittest.main()
