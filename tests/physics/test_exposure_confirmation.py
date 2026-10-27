import unittest

from embodied_agent.exposure_confirmation import ExposureClockMap, ExposureConfirmation
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.release_confirmation import ReleaseConfirmation


class ExposureConfirmationTests(unittest.TestCase):
    def test_delivery_time_can_hide_motion_but_exposure_time_does_not(self):
        legacy = ReleaseConfirmation(0)
        check = ExposureConfirmation(0, ExposureClockMap(100, 0))
        for i in range(7):
            delivery, exposure = i * 0.2, i * 0.16
            point = (0.62 + 0.017 * delivery, 0, 0.04)
            legacy.observe(delivery, point)
            check.step(delivery, 100 + delivery, StampedPosition(point, 100 + exposure))
        self.assertEqual(legacy.status, "confirmed_in_tray")
        self.assertEqual(check.status, "pending")
        self.assertEqual(check.step(3, 103, None), "unconfirmed")

    def test_stable_position_requires_one_second_of_exposure_evidence(self):
        check = ExposureConfirmation(0, ExposureClockMap(100, 0))
        for i in range(7):
            delivery = 0.2 * i
            exposure = max(0, delivery - 0.1)
            check.step(delivery, 100 + delivery, StampedPosition((0.64, 0, 0.04), 100 + exposure))
            if i < 6:
                self.assertEqual(check.status, "pending")
        self.assertEqual(check.status, "confirmed_in_tray")

    def test_drift_pause_future_age_and_bad_mapping_rejected(self):
        mapping = ExposureClockMap(100, 0)
        for captured, host, task in [
            (100.1, 100.2, 0.1),
            (100.3, 100.2, 0.2),
            (100, 100.3, 0.3),
            (float("nan"), 100, 0),
        ]:
            with self.assertRaises(ValueError):
                mapping.map(captured, host, task)
        for rate in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                ExposureClockMap(100, 0, rate=rate)

    def test_rejected_observation_breaks_stable_chain(self):
        check = ExposureConfirmation(0, ExposureClockMap(100, 0))
        for i in range(6):
            t = i / 5
            host = 100 + t
            observation = StampedPosition((0.64, 0, 0.04), host if i != 3 else 99)
            check.step(t, host, observation)
        self.assertEqual(check.status, "pending")

    def test_deadline_and_stop_use_delivery_clock(self):
        for stop, expected in [(False, "unconfirmed"), (True, "cancelled")]:
            check = ExposureConfirmation(0, ExposureClockMap(100, 0))
            check.step(3.1, 103.1, StampedPosition((0.64, 0, 0.04), 102.9), stop=stop)
            self.assertEqual(check.status, expected)
        check = ExposureConfirmation(0, ExposureClockMap(100, 0))
        check.step(3, 103, StampedPosition((0.70, 0, 0.04), 102.9))
        self.assertEqual(check.status, "outside_confirmation_region")

    def test_mapping_supports_explicit_rate_and_rejects_repeated_exposure(self):
        mapping = ExposureClockMap(100, 0, rate=2)
        self.assertAlmostEqual(mapping.map(100.1, 100.2, 0.4), 0.2)
        check = ExposureConfirmation(0, ExposureClockMap(100, 0))
        check.step(0, 100, StampedPosition((0.64, 0, 0.04), 100))
        check.step(0.2, 100.2, StampedPosition((0.64, 0, 0.04), 100))
        self.assertIn("fresh", check.last_error)
