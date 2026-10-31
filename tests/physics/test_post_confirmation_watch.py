import unittest

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.post_confirmation_watch import PostConfirmationWatch
from scripts.check_post_confirmation_watch import scenarios


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.original = {"status": "confirmed_in_tray", "reason": "stable_visual_evidence"}
        self.watch = PostConfirmationWatch(self.original, 1, "camera", ExposureClockMap(16, 0))

    def step(self, task, point=(0.64, 0, 0.04), *, exposure=None, session="camera"):
        frame = (
            None
            if point is None
            else StampedPosition(point, 16 + (task if exposure is None else exposure))
        )
        return self.watch.step(task + 0.015625, 16 + task + 0.015625, frame, session_id=session)

    def test_sustained_outside_alert_preserves_original_confirmation(self):
        self.step(1.25, (0.7, 0, 0.04))
        self.assertEqual(self.watch.evidence_state, "outside_candidate")
        self.assertFalse(self.watch.result()["outside_detected"])
        self.step(1.5, (0.7, 0, 0.04))
        self.assertTrue(self.watch.result()["outside_detected"])
        self.assertEqual(self.watch.evidence_state, "outside_region")
        self.assertEqual(self.original["status"], "confirmed_in_tray")
        self.step(1.75)
        self.assertEqual(self.watch.evidence_state, "inside_region")
        self.assertTrue(self.watch.result()["outside_detected"])
        self.step(2, (0.7, 0, 0.04))
        self.step(2.25, (0.7, 0, 0.04))
        self.assertEqual(len(self.watch.alerts), 1)

    def test_transient_outside_does_not_trigger_alert(self):
        self.step(1.25, (0.7, 0, 0.04))
        self.step(1.5)
        self.step(1.75, (0.7, 0, 0.04))
        self.assertFalse(self.watch.result()["outside_detected"])

    def test_missing_and_long_gap_break_outside_continuity(self):
        self.step(1.25, (0.7, 0, 0.04))
        self.step(1.5, None)
        self.assertEqual(self.watch.evidence_state, "unknown")
        self.step(1.75, (0.7, 0, 0.04))
        self.assertFalse(self.watch.result()["outside_detected"])
        self.step(2.25, (0.7, 0, 0.04))
        self.assertFalse(self.watch.result()["outside_detected"])
        self.step(2.5, (0.7, 0, 0.04))
        self.assertTrue(self.watch.result()["outside_detected"])

    def test_stale_future_and_pre_confirmation_frames_are_unknown(self):
        self.step(1.25, exposure=1)
        self.assertEqual(self.watch.reason, "observation_rejected")
        self.step(1.5)
        self.step(1.75, exposure=1.5)
        self.assertEqual(self.watch.reason, "observation_rejected")
        self.step(2, exposure=2.5)
        self.assertEqual(self.watch.evidence_state, "unknown")
        self.assertFalse(self.watch.alerts)

    def test_session_change_invalidates_and_stop_has_priority(self):
        self.step(1.25)
        self.assertEqual(self.step(1.5, session="replacement"), "invalidated")
        self.step(1.75)
        self.assertEqual(self.watch.evidence_state, "unknown")
        self.setUp()
        self.watch.step(float("nan"), float("nan"), None, session_id="replacement", stop=True)
        self.assertEqual(self.watch.status, "cancelled")
        self.assertEqual(self.watch.reason, "user_stop")

    def test_deadline_cannot_reuse_inside_state_or_consume_frame(self):
        self.step(1.25)
        self.watch.step(3, 19, StampedPosition((0.7, 0, 0.04), 19), session_id="camera")
        self.assertEqual(self.watch.status, "completed")
        self.assertEqual(self.watch.evidence_state, "unknown")
        self.assertFalse(self.watch.alerts)
        self.assertEqual(self.watch.last_exposure, 1.25)

    def test_invalid_inputs_and_mapping_drift_fail_closed(self):
        for result, duration in (
            ({"status": "pending"}, 2),
            (self.original, 0),
            (self.original, float("nan")),
        ):
            with self.assertRaises(ValueError):
                PostConfirmationWatch(
                    result, 1, "camera", ExposureClockMap(16, 0), duration=duration
                )
        self.step(1.25, (float("nan"), 0, 0.04))
        self.assertEqual(self.watch.evidence_state, "unknown")
        self.watch.step(1.5, 18, StampedPosition((0.64, 0, 0.04), 17.5), session_id="camera")
        self.assertEqual(self.watch.reason, "observation_rejected")
        with self.assertRaises(ValueError):
            self.step(1.25)

    def test_result_and_original_are_detached(self):
        self.original["status"] = "changed"
        result = self.watch.result()
        result["original_confirmation"]["status"] = "changed_again"
        result["alerts"].append({"event": "fake"})
        self.assertEqual(
            self.watch.result()["original_confirmation"]["status"], "confirmed_in_tray"
        )
        self.assertEqual(self.watch.alerts, [])

    def test_scenarios_start_with_real_confirmation_and_separate_followup(self):
        report = scenarios()
        self.assertEqual(report["original_confirmation"]["status"], "confirmed_in_tray")
        for name, case in report["cases"].items():
            self.assertEqual(
                case["final"]["outside_detected"], name in {"outside", "return_inside"}
            )
            self.assertEqual(
                case["final"]["original_confirmation"], report["original_confirmation"]
            )
        self.assertEqual(report["cases"]["session_change"]["final"]["status"], "invalidated")
        self.assertEqual(report["cases"]["stop"]["final"]["status"], "cancelled")
        self.assertEqual(report["cases"]["missing"]["before_end"]["evidence_state"], "unknown")
