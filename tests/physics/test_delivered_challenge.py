import unittest

from embodied_agent.delivered_challenge import DeliveredChallenge
from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from scripts.evaluate_delivered_challenge import replay

P = (0.64, 0, 0.045)


def create():
    return DeliveredChallenge(0, ExposureClockMap(100, 0), "camera", position_error_m=(0.001,) * 3)


def send(check, t, delay=0, **kwargs):
    point = StampedPosition(P, 100 + t)
    host = 100 + t + delay
    return check.deliver(
        host - 100, host, point, point, session_id=kwargs.get("session_id", "camera")
    )


class DeliveredChallengeTests(unittest.TestCase):
    def test_zero_latency_and_nonzero_clock_origin(self):
        check = create()
        while check.next_capture_time is not None:
            send(check, check.next_capture_time)
        self.assertEqual(check.status, "confirmed_in_tray")
        self.assertEqual(check.result()["evidence"]["captures"], 31)

    def test_positive_latency_cannot_extend_deadline(self):
        check = create()
        while check.next_capture_time is not None:
            send(check, check.next_capture_time, 0.02)
        self.assertEqual(check.status, "unconfirmed")
        self.assertEqual(check.reason, "delivery_deadline_exceeded")
        self.assertNotEqual(check.evidence.status, "confirmed_in_tray")

    def test_watchdog_without_delivery(self):
        check = create()
        check.tick(3)
        self.assertEqual(check.status, "unconfirmed")
        send(check, 3)
        self.assertEqual(check.status, "unconfirmed")

    def test_duplicate_exposure_and_session(self):
        check = create()
        send(check, 0)
        send(check, 0, 0.01)
        self.assertEqual(check.reason, "exposure_rejected")
        check = create()
        send(check, 0, session_id="other")
        self.assertEqual(check.reason, "camera_session_changed")

    def test_skew_missing_stale_future_and_drift(self):
        cases = [
            (0.01, 100.01, StampedPosition(P, 100), StampedPosition(P, 100.001)),
            (0, 100, None, None),
            (0.3, 100.3, StampedPosition(P, 100), StampedPosition(P, 100)),
            (0, 100, StampedPosition(P, 101), StampedPosition(P, 101)),
            (0.1, 100, StampedPosition(P, 100), StampedPosition(P, 100)),
        ]
        for t, h, a, b in cases:
            check = create()
            check.deliver(t, h, a, b, session_id="camera")
            self.assertEqual(check.reason, "exposure_rejected")

    def test_task_order_invalid_without_mutation(self):
        check = create()
        check.tick(0.1)
        before = check.result()
        with self.assertRaises(ValueError):
            send(check, 0)
        self.assertEqual(check.result(), before)

    def test_replay_latency_and_watchdog(self):
        times = [i * 0.2 for i in range(11)] + [2 + i * 0.05 for i in range(1, 21)]
        case = {
            "samples": [
                {
                    "elapsed_s": t,
                    "recipients": ["challenge"],
                    "profiles": {"coverage": {"primary": P, "secondary": P}},
                }
                for t in times
            ]
        }
        result = replay(case, (0.001,) * 3, "final_late_20ms")
        self.assertEqual(result["raw_exposure_only"]["status"], "confirmed_in_tray")
        self.assertEqual(result["guarded"]["status"], "unconfirmed")
        self.assertEqual(result["guarded"]["reason"], "delivery_deadline_without_complete_evidence")
        result = replay(case, (0.001,) * 3, "drop_final")
        self.assertEqual(result["raw_exposure_only"]["status"], "pending")
        self.assertEqual(result["guarded"]["status"], "unconfirmed")

    def test_invalid_constructor_and_equal_delivery_time(self):
        with self.assertRaises(TypeError):
            DeliveredChallenge(0, None, "camera", position_error_m=(0.001,) * 3)
        with self.assertRaises(ValueError):
            DeliveredChallenge(0, ExposureClockMap(0, 0), "", position_error_m=(0.001,) * 3)
        check = create()
        send(check, 0)
        before = check.result()
        with self.assertRaises(ValueError):
            send(check, 0)
        self.assertEqual(check.result(), before)

    def test_stop_and_snapshot(self):
        check = create()
        send(check, 0)
        result = check.result()
        result["deliveries"].clear()
        self.assertEqual(len(check.result()["deliveries"]), 1)
        check.tick(0.1, stop=True)
        self.assertEqual(check.status, "cancelled")
        self.assertIsNone(check.next_capture_time)
