import unittest

from scripts.check_exposure_pipeline import run_case


class ExposureStressTests(unittest.TestCase):
    def test_jitter_and_early_drop_recover_without_false_motion_confirmation(self):
        for mode in ("jitter", "early_drop"):
            row = run_case(mode)
            self.assertEqual(row["status"], "confirmed_in_tray")
            confirmed = next(o for o in row["observations"] if o["status"] == "confirmed_in_tray")
            self.assertGreaterEqual(confirmed["task_time"], 2 if mode == "early_drop" else 1)
        self.assertNotEqual(run_case("moving_jitter")["status"], "confirmed_in_tray")

    def test_gaps_and_drops_cannot_stitch_discontinuous_evidence(self):
        for mode in ("exposure_gaps", "periodic_drop", "all_drop"):
            row = run_case(mode)
            self.assertEqual(row["status"], "unconfirmed")
            self.assertEqual(len(row["calls"]), 15)
            self.assertFalse(any(o["status"] == "confirmed_in_tray" for o in row["observations"]))

    def test_clock_recovery_needs_new_evidence_and_late_poll_cannot_extend_budget(self):
        self.assertEqual(run_case("clock_jump")["status"], "unconfirmed")
        recovered = run_case("clock_recovery")
        self.assertEqual(recovered["status"], "confirmed_in_tray")
        rejected = [o for o in recovered["observations"] if o["last_error"]]
        self.assertTrue(rejected)
        confirmed = next(o for o in recovered["observations"] if o["status"] == "confirmed_in_tray")
        self.assertGreater(confirmed["task_time"] - rejected[-1]["task_time"], 1)
        late = run_case("late_poll")
        self.assertEqual(late["status"], "unconfirmed")
        self.assertEqual(len(late["calls"]), 3)
        self.assertEqual(late["observations"][-1]["reason"], "delivery_deadline_exceeded")
