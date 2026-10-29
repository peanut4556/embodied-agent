import json
import tempfile
import unittest
from pathlib import Path

from embodied_agent.recovery_exposure_observer import RecoveryExposureObserver
from embodied_agent.recovery_report import recovery_report, write_recovery_report
from scripts.check_recovery_reports import ScenarioSession, scenarios


class RecoveryReportTests(unittest.TestCase):
    def test_restored_observation_is_not_confirmation(self):
        cases = scenarios()["cases"]
        pending = cases["restored_pending"]["before_close"]["diagnosis"]
        confirmed = cases["confirmed"]["before_close"]["diagnosis"]
        self.assertTrue(pending["observation_restored"])
        self.assertFalse(pending["visual_confirmation_succeeded"])
        self.assertFalse(pending["terminal"])
        self.assertTrue(confirmed["visual_confirmation_succeeded"])
        self.assertTrue(confirmed["cleanup_required"])
        self.assertFalse(cases["confirmed"]["after_close"]["diagnosis"]["cleanup_required"])
        self.assertTrue(
            cases["confirmed"]["after_close"]["diagnosis"]["visual_confirmation_succeeded"]
        )

    def test_distinct_failure_reasons_and_counts(self):
        cases = scenarios()["cases"]
        expected = {
            "limit_reached": ("restart_limit_reached", 1),
            "budget_rejected": ("insufficient_recovery_budget", 0),
            "restart_failed": ("camera_restart_failed", 1),
            "stopped": ("user_stop", 1),
        }
        for name, (reason, count) in expected.items():
            report = cases[name]["before_close"]
            self.assertEqual(report["diagnosis"]["reason_code"], reason)
            self.assertEqual(report["result"]["restart_count"], count)
            self.assertFalse(report["diagnosis"]["visual_confirmation_succeeded"])

    def test_cleanup_failure_is_not_reported_as_successful_close(self):
        case = scenarios()["cases"]["cleanup_failed"]
        report = case["after_close"]
        self.assertEqual(report["cleanup"]["status"], "failed")
        self.assertEqual(report["cleanup"]["error"], case["cleanup_exception"])
        self.assertTrue(report["diagnosis"]["cleanup_required"])
        self.assertEqual(report["result"]["recovery_attempts"][0]["outcome"], "cleanup_failed")

    def test_snapshot_is_detached_and_does_not_control_camera(self):
        session = ScenarioSession()
        observer = RecoveryExposureObserver(session, 0, task_clock=lambda: 0.5)
        observer.restart()
        session.status = "failed"
        observer.step(0.6)
        snapshot = recovery_report(observer)
        snapshot["events"][0]["event"] = "tampered"
        snapshot["cleanup"]["status"] = "tampered"
        snapshot["result"]["recovery_attempts"][0]["outcome"] = "tampered"
        self.assertNotEqual(observer.rows[0]["event"], "tampered")
        self.assertEqual(observer.cleanup["status"], "not_requested")
        self.assertEqual(session.status, "failed")
        self.assertEqual(observer.attempts[0]["outcome"], "camera_failed_again")
        observer.close()
        self.assertEqual(snapshot["camera"]["status"], "failed")

    def test_export_round_trip_and_refuses_overwrite(self):
        report = scenarios()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_recovery_report(path, report)
            self.assertEqual(json.loads(path.read_text()), report)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_recovery_report(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), original)

    def test_nonfinite_export_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            with self.assertRaises(ValueError):
                write_recovery_report(path, {"task_time": float("nan")})
            self.assertFalse(path.exists())

    def test_cleanup_retry_updates_current_state(self):
        session = ScenarioSession()
        observer = RecoveryExposureObserver(session, 0, task_clock=lambda: 0.5)
        observer.restart()
        session.fail_close = True
        with self.assertRaises(RuntimeError):
            observer.close()
        failed = recovery_report(observer)
        session.fail_close = False
        observer.close()
        self.assertEqual(observer.cleanup["status"], "succeeded")
        self.assertEqual(failed["cleanup"]["status"], "failed")
        self.assertEqual(observer.attempts[0]["outcome"], "cleanup_failed")

    def test_visual_success_does_not_hide_cleanup_failure(self):
        report = scenarios()["cases"]["confirmed_cleanup_failed"]["after_close"]
        self.assertTrue(report["diagnosis"]["visual_confirmation_succeeded"])
        self.assertTrue(report["diagnosis"]["observation_restored"])
        self.assertTrue(report["diagnosis"]["cleanup_required"])
        self.assertEqual(report["cleanup"]["status"], "failed")
