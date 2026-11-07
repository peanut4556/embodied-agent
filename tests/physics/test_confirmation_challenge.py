import unittest

from embodied_agent.confirmation_challenge import ConfirmationChallenge

P = (0.64, 0, 0.045)
B = (0.001,) * 3


def candidate():
    check = ConfirmationChallenge(0, position_error_m=B)
    for i in range(11):
        check.step(i * 0.2, P, P)
    return check


class ConfirmationChallengeTests(unittest.TestCase):
    def test_candidate_withheld_and_exact_budget(self):
        check = candidate()
        self.assertEqual(check.status, "pending")
        self.assertEqual(check.phase, "challenge")
        self.assertAlmostEqual(check.next_capture_time, 2.05)
        for i in range(1, 21):
            check.step(2 + i * 0.05, P, P)
        self.assertEqual(check.status, "confirmed_in_tray")
        self.assertEqual(check.result()["captures"], 31)
        self.assertIsNone(check.next_capture_time)

    def test_secondary_motion_veto(self):
        check = candidate()
        check.step(2.05, P, (0.642, 0, 0.045))
        self.assertEqual(check.reason, "challenge_motion")
        self.assertEqual(check.status, "unconfirmed")

    def test_missing_boundary_and_conflict(self):
        for point in (None, (0.69, 0, 0.045), (0.66, 0, 0.045)):
            check = candidate()
            check.step(2.05, P, point)
            self.assertEqual(check.reason, "challenge_view_rejected")

    def test_gap_and_deadline(self):
        check = candidate()
        check.step(2.08, P, P)
        self.assertEqual(check.reason, "challenge_capture_gap")
        check = candidate()
        check.step(3.01, P, P)
        self.assertEqual(check.reason, "deadline_exceeded")

    def test_invalid_timestamp_does_not_mutate(self):
        check = candidate()
        before = check.result()
        for timestamp in (2, 2.01, float("nan")):
            with self.assertRaises(ValueError):
                check.step(timestamp, P, P)
            self.assertEqual(check.result(), before)

    def test_late_candidate_cannot_extend_deadline(self):
        check = ConfirmationChallenge(0, position_error_m=B)
        for i in range(12):
            p = None if i == 0 else P
            check.step(i * 0.2, p, p)
        self.assertEqual(check.reason, "insufficient_challenge_time")
        self.assertEqual(check.status, "unconfirmed")

    def test_cancel_snapshot_and_terminal(self):
        check = candidate()
        snap = check.result()
        snap["observations"].clear()
        self.assertEqual(len(check.result()["observations"]), 11)
        check.step(2.05, None, None, stop=True)
        self.assertEqual(check.status, "cancelled")
        check.step(3, P, P)
        self.assertEqual(check.status, "cancelled")

    def test_delayed_burst_cannot_pass_with_too_few_frames(self):
        check = candidate()
        for i in range(1, 19):
            check.step(2 + i * 0.055, P, P)
        check.step(3, P, P)
        self.assertEqual(check.status, "unconfirmed")
        self.assertEqual(check.reason, "challenge_budget_exhausted")
        self.assertLess(check.result()["captures"], 31)

    def test_no_candidate_deadline(self):
        check = ConfirmationChallenge(0, position_error_m=B)
        for i in range(16):
            check.step(i * 0.2, None, None)
        self.assertEqual(check.status, "unconfirmed")
        self.assertEqual(check.reason, "no_candidate")
