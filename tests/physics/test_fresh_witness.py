import unittest

from embodied_agent.fresh_witness import FreshWitnessGate
from embodied_agent.observation_capture import StampedPosition
from scripts.evaluate_sampling_aliases import run

P = (0.64, 0, 0.045)


def create():
    holders = []
    run("sweep", None, 0, None, jitter_max_ms=2, on_controller=holders.append)
    return FreshWitnessGate(holders[0])


def deliver(g, offset, age=0.01, position=P, session="audit"):
    t = g.start + offset
    p = StampedPosition(position, t)
    return g.deliver(t + age, t + age, p, p, session_id=session)


class FreshWitnessTests(unittest.TestCase):
    def test_five_new_exposures_required_and_temporal_scope(self):
        g = create()
        for i in range(4):
            self.assertEqual(deliver(g, i * 0.01), "pending")
        self.assertEqual(deliver(g, 0.04), "confirmed_recent_evidence")
        r = g.result()
        self.assertAlmostEqual(r["last_exposure_s"] - r["first_exposure_s"], 0.04)
        self.assertAlmostEqual(r["decision_time_s"] - r["last_exposure_s"], 0.01)
        self.assertFalse(r["guarantees_current_or_future_stability"])

    def test_slow_path_is_not_a_replay_workaround(self):
        g = create()
        self.assertEqual(deliver(g, 0, 0.08), "unconfirmed")
        self.assertEqual(g.reason, "stale_witness")

    def test_anchor_detects_stationary_but_displaced_witness(self):
        g = create()
        deliver(g, 0, position=(0.642, 0, 0.045))
        self.assertEqual(g.reason, "witness_motion")

    def test_cached_duplicate_and_gap_fail(self):
        g = create()
        p = StampedPosition(P, g.start - 0.001)
        g.deliver(g.start + 0.01, g.start + 0.01, p, p, session_id="audit")
        self.assertEqual(g.reason, "pre_candidate_exposure")
        for offset in (0, 0.04):
            g = create()
            deliver(g, 0)
            t = g.start + offset
            p = StampedPosition(P, t)
            delivery = g.start + (0.02 if offset == 0 else 0.05)
            g.deliver(delivery, delivery, p, p, session_id="audit")
            self.assertEqual(g.reason, "witness_capture_gap")

    def test_missing_session_and_view_conflict(self):
        g = create()
        deliver(g, 0, session="other")
        self.assertEqual(g.reason, "camera_session_changed")
        g = create()
        g.deliver(g.start + 0.01, g.start + 0.01, None, None, session_id="audit")
        self.assertEqual(g.reason, "missing_timestamped_pair")
        g = create()
        a = StampedPosition(P, g.start)
        b = StampedPosition((0.66, 0, 0.045), g.start)
        g.deliver(g.start + 0.01, g.start + 0.01, a, b, session_id="audit")
        self.assertEqual(g.reason, "witness_view_rejected")

    def test_cancel_timeout_and_terminal_latch(self):
        for stop in (True, False):
            g = create()
            g.tick(g.deadline, stop=stop)
            before = g.result()
            deliver(g, 0)
            self.assertEqual(before, g.result())
            self.assertEqual(g.reason, "user_stop" if stop else "witness_deadline")

    def test_boundary_delivery_and_snapshot(self):
        g = create()
        for i in range(5):
            deliver(g, i * 0.02, age=0.02)
        self.assertEqual(g.status, "confirmed_recent_evidence")
        r = g.result()
        r["observations"].clear()
        self.assertEqual(len(g.rows), 5)

    def test_clock_backwards_is_nonmutating(self):
        g = create()
        deliver(g, 0)
        before = g.result()
        with self.assertRaises(ValueError):
            g.tick(g.start)
        self.assertEqual(before, g.result())

    def test_candidate_required(self):
        with self.assertRaises(ValueError):
            FreshWitnessGate({"status": "confirmed_in_tray"})
