import unittest

from embodied_agent.active_view import ActiveView
from embodied_agent.release_observer import ReleaseObserver


class ActiveViewTests(unittest.TestCase):
    def test_missing_primary_switches_once_without_eager_secondary_read(self):
        selector = ActiveView()
        calls = []

        def locate(view):
            calls.append(view)

        for timestamp in (0, 0.2, 0.4, 0.6, 0.8):
            self.assertIsNone(selector.observe(timestamp, locate))
        self.assertEqual(calls, ["primary", "primary", "secondary", "secondary", "secondary"])
        self.assertEqual(len(selector.result()["switches"]), 1)

    def test_good_detection_resets_consecutive_failure_count(self):
        selector = ActiveView()
        for timestamp, point in ((0, None), (0.2, (0.64, 0, 0.04)), (0.4, None)):
            selector.observe(timestamp, lambda view, value=point: value)
        self.assertEqual(selector.view, "primary")
        self.assertEqual(selector.result()["switches"], [])

    def test_outside_detection_is_not_hidden_by_trying_another_view(self):
        selector = ActiveView()
        for timestamp in (0, 0.2, 0.4):
            self.assertEqual(
                selector.observe(timestamp, lambda view: (0.8, 0, 0.04)), (0.8, 0, 0.04)
            )
        self.assertEqual(selector.view, "primary")

    def test_invalid_detections_and_errors_are_recorded(self):
        for bad in ((float("nan"), 0, 0), (1, 2), (10**400, 0, 0)):
            selector = ActiveView()
            self.assertIsNone(selector.observe(0, lambda view, value=bad: value))
            self.assertIsNotNone(selector.result()["observations"][0]["error"])

        def fail(view):
            raise ValueError("red block not visible")

        selector.observe(0.2, fail)
        self.assertEqual(selector.view, "secondary")

    def test_switch_breaks_temporal_evidence_without_resetting_deadline(self):
        selector, observer = ActiveView(), ReleaseObserver()
        release = [{"event": "intentional_release"}]
        for tick in range(16):
            timestamp = tick * 0.2

            def locate(view, current=tick):
                return None if current in (3, 4) else (0.64, 0, 0.04)

            observer.step(
                timestamp, release, lambda now=timestamp, read=locate: selector.observe(now, read)
            )
            if tick == 9:  # New evidence starts at 1.0, not at original 0.0.
                self.assertEqual(observer.status, "pending")
        self.assertEqual(observer.status, "confirmed_in_tray")
        self.assertEqual(observer.confirmation.start, 0)
        self.assertEqual(len(selector.switches), 1)

    def test_all_missing_times_out_and_stop_prevents_extra_capture(self):
        selector, observer = ActiveView(), ReleaseObserver()
        for tick in range(16):
            timestamp = tick * 0.2
            observer.step(
                timestamp,
                [{"event": "intentional_release"}],
                lambda now=timestamp: selector.observe(now, lambda view: None),
            )
        self.assertEqual(observer.status, "unconfirmed")
        size = len(selector.rows)
        observer.step(4, [], lambda: self.fail("terminal state read camera"))
        self.assertEqual(len(selector.rows), size)
        observer = ReleaseObserver()
        observer.step(0, [], lambda: self.fail("stop read camera"), stop=True)
        self.assertEqual(observer.status, "cancelled")

    def test_late_switch_cannot_extend_confirmation_budget(self):
        selector, observer = ActiveView(), ReleaseObserver()
        for tick in range(16):
            timestamp = tick * 0.2

            def locate(view, current=tick):
                if view == "secondary":
                    return (0.64, 0, 0.04)
                return None if current >= 10 else (0.8, 0, 0.04)

            observer.step(
                timestamp,
                [{"event": "intentional_release"}],
                lambda now=timestamp, read=locate: selector.observe(now, read),
            )
        self.assertEqual(selector.view, "secondary")
        self.assertEqual(observer.status, "unconfirmed")
        self.assertEqual(observer.confirmation.start, 0)

    def test_invalid_time_has_no_capture_side_effect_and_result_is_detached(self):
        selector = ActiveView()
        selector.observe(1, lambda view: None)
        for timestamp in (1, 0, float("nan"), True):
            with self.assertRaises(ValueError):
                selector.observe(timestamp, lambda view: self.fail("invalid time read camera"))
        result = selector.result()
        result["observations"].clear()
        self.assertEqual(len(selector.rows), 1)
