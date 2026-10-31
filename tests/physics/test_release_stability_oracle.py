import json
import unittest
from pathlib import Path

from embodied_agent.exposure_confirmation import ExposureClockMap, ExposureConfirmation
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.recovery_replay import replay_report
from embodied_agent.release_confirmation import ReleaseConfirmation
from scripts.audit_release_stability import evaluate, reference_first_confirmation


class StabilityOracleTests(unittest.TestCase):
    def test_seeded_differential_matrix(self):
        report = evaluate()
        self.assertEqual(report["cases"], 413)
        self.assertEqual(
            report["mismatches"], 0, [r["case"] for r in report["results"] if not r["match"]]
        )

    def test_boundary_entry_cannot_borrow_outside_exposure(self):
        monitor = ReleaseConfirmation(0)
        for i in range(6):
            monitor.observe(i * 0.25, (0.64, 0, 0.032 if i == 0 else 0.033))
            self.assertEqual(monitor.status, "pending" if i < 5 else "confirmed_in_tray")
        self.assertEqual(monitor.since, 0.25)

    def test_dual_clock_inherits_corrected_window(self):
        monitor = ExposureConfirmation(0, ExposureClockMap(10, 0))
        for i in range(6):
            task = i * 0.25
            monitor.step(
                task + 0.01,
                10 + task + 0.01,
                StampedPosition((0.64, 0, 0.032 if i == 0 else 0.033), 10 + task),
            )
            self.assertEqual(monitor.status, "pending" if i < 5 else "confirmed_in_tray")

    def test_reference_missing_frame_and_gap_break_windows(self):
        samples = [(i * 0.25, [0.64, 0, 0.04]) for i in range(6)]
        self.assertEqual(reference_first_confirmation(samples), 1)
        samples[2] = (0.5, None)
        self.assertIsNone(reference_first_confirmation(samples))
        samples.pop(2)
        self.assertIsNone(reference_first_confirmation(samples))

    def test_reference_speed_and_deadline(self):
        moving = [(i * 0.25, [0.62 + 0.021 * i * 0.25, 0, 0.04]) for i in range(9)]
        self.assertIsNone(reference_first_confirmation(moving))
        late = [(2.25 + i * 0.25, [0.64, 0, 0.04]) for i in range(5)]
        self.assertIsNone(reference_first_confirmation(late))

    def test_v1_evidence_not_silently_replayed_with_v2(self):
        root = Path(__file__).resolve().parents[2]
        report = json.loads((root / "docs/evaluations/recovery-evidence-v1.json").read_text())
        for case in report["cases"].values():
            replay = replay_report(case["after_close"])
            self.assertEqual(
                replay, {"status": "unavailable", "reason": "superseded_confirmation_algorithm"}
            )
