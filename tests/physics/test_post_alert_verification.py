import unittest

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.post_alert_verification import PostAlertVerification
from scripts.evaluate_post_alert_verification import evaluate


class PostAlertVerificationTests(unittest.TestCase):
    def setUp(self):
        self.original = {"status": "confirmed_in_tray"}
        self.subject = PostAlertVerification(self.original, 1, "camera", ExposureClockMap(16, 0))

    def step(self, task, point=(0.64, 0, 0.04), *, exposure=None):
        frame = (
            None
            if point is None
            else StampedPosition(point, 16 + (task if exposure is None else exposure))
        )
        return self.subject.step(task, 16 + task, frame, session_id="camera")

    def alert(self):
        self.step(1.25, (0.7, 0, 0.04))
        self.step(1.5, (0.7, 0, 0.04))
        self.assertEqual(self.subject.status, "rechecking")

    def test_new_stable_window_reconfirms_and_preserves_alert_and_original(self):
        self.alert()
        for tick in range(8):
            self.assertEqual(self.step(1.625 + tick / 8), "rechecking")
        self.assertEqual(self.step(2.625), "reconfirmed")
        result = self.subject.result()
        self.assertEqual(result["original_confirmation"], self.original)
        self.assertIsNotNone(result["alert"])
        self.assertEqual(result["recheck_start"], 1.5)
        self.assertEqual(result["deadline"], 4.5)
        self.assertEqual([row["status"] for row in result["events"]], ["rechecking", "reconfirmed"])

    def test_single_frame_return_does_not_clear_and_deadline_never_extends(self):
        self.alert()
        self.step(1.625)
        for tick in range(14, 37):
            self.step(tick / 8, (0.7, 0, 0.04))
        self.assertEqual(self.subject.status, "needs_review")
        self.assertEqual(self.subject.result()["deadline"], 4.5)

    def test_missing_and_duplicate_frames_break_stability(self):
        for missing in (True, False):
            self.setUp()
            self.alert()
            for tick in range(13, 20):
                self.step(tick / 8)
            self.step(2.5, None if missing else (0.64, 0, 0.04), exposure=2.375)
            self.step(2.625)
            self.assertEqual(self.subject.status, "rechecking")
            for tick in range(22, 30):
                self.step(tick / 8)
            self.assertEqual(self.subject.status, "reconfirmed")

    def test_delayed_pre_alert_frame_cannot_seed_recheck(self):
        self.alert()
        self.step(1.625, exposure=1.5)
        self.assertIsNone(self.subject.result()["recheck"]["last_exposure_time"])
        for tick in range(14, 22):
            self.step(tick / 8)
        self.assertEqual(self.subject.status, "rechecking")
        self.step(2.75)
        self.assertEqual(self.subject.status, "reconfirmed")

    def test_late_complete_window_cannot_beat_delivery_deadline(self):
        self.alert()
        for tick in range(29, 37):
            self.step(tick / 8)
        self.assertEqual(self.subject.status, "needs_review")
        self.step(4.625)
        self.assertEqual(self.subject.status, "needs_review")

    def test_clocks_and_session_fail_closed_without_observations(self):
        for task, host, session in (
            (1.625, 18, "camera"),
            (1.625, float("nan"), "camera"),
            (1.5, 17.5, "camera"),
            (1.625, 17.625, "replacement"),
        ):
            self.setUp()
            self.alert()
            self.subject.step(task, host, None, session_id=session)
            self.assertEqual(self.subject.status, "invalidated")

    def test_stop_precedes_invalid_inputs_and_terminal_state_is_latched(self):
        self.alert()
        self.subject.step(None, None, None, session_id=None, stop=True)
        self.assertEqual(self.subject.status, "cancelled")
        before = self.subject.result()
        self.step(2)
        self.assertEqual(self.subject.result(), before)

    def test_no_alert_is_not_reconfirmation_including_all_missing(self):
        for point in (None, (0.64, 0, 0.04)):
            self.setUp()
            for tick in range(9, 25):
                self.step(tick / 8, point)
            self.assertEqual(self.subject.status, "completed_without_alert")
            self.assertIsNone(self.subject.result()["recheck"])

    def test_snapshots_do_not_mutate_state(self):
        self.alert()
        self.original["status"] = "modified"
        result = self.subject.result()
        result["events"].clear()
        result["alert"]["event"] = "modified"
        self.assertEqual(len(self.subject.result()["events"]), 1)
        self.assertEqual(
            self.subject.result()["original_confirmation"]["status"], "confirmed_in_tray"
        )

    def test_malformed_frames_break_recheck_without_crashing(self):
        for frame in (
            object(),
            StampedPosition((0.64, 0), 18),
            StampedPosition((float("nan"), 0, 0.04), 18),
            StampedPosition((10**400, 0, 0.04), 18),
        ):
            self.setUp()
            self.alert()
            self.step(1.75)
            self.subject.step(2, 18, frame, session_id="camera")
            self.assertEqual(self.subject.status, "rechecking")
            self.assertIsNotNone(self.subject.result()["last_error"])
            for tick in range(17, 25):
                self.step(tick / 8)
            self.assertEqual(self.subject.status, "rechecking")
            self.step(3.125)
            self.assertEqual(self.subject.status, "reconfirmed")

    def test_72_scenarios_and_single_frame_counterexamples(self):
        report = evaluate()
        self.assertEqual(report["cases"], 72)
        self.assertEqual(report["mismatches"], 0)
        self.assertEqual(
            report["status_counts"],
            {
                "reconfirmed": 9,
                "needs_review": 36,
                "invalidated": 18,
                "cancelled": 9,
            },
        )
        self.assertEqual(report["single_frame_premature_clears"], 18)
