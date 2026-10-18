import threading
import unittest
from unittest.mock import patch

from embodied_agent.sim_app import SimulationApp


class StopConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.app = SimulationApp()
        self.app.run_id = "old-run"
        self.app.task.update(run_id="old-run", phase="stopped")

    def test_failed_stop_is_unconfirmed_and_can_be_retried(self):
        with (
            patch("embodied_agent.sim_app.bridge", side_effect=OSError("offline")),
            self.assertRaises(OSError),
        ):
            self.app.stop()
        self.assertTrue(self.app.cancel.is_set())
        self.assertEqual(self.app.task["stop_confirmation"], "unconfirmed")
        self.assertIn("尚未确认", self.app.task["message"])
        self.assertEqual(self.app.task["stop_error"], "offline")
        with patch("embodied_agent.sim_app.bridge", return_value={"success": True}):
            self.app.stop()
        self.assertEqual(self.app.task["stop_confirmation"], "confirmed")
        self.assertEqual(self.app.task["stop_error"], "")
        self.assertIn("已确认停止", self.app.task["message"])

    def test_late_failure_cannot_erase_confirmation(self):
        with patch("embodied_agent.sim_app.bridge", return_value={"success": True}):
            self.app._stop_execution("old-run")
        with (
            patch("embodied_agent.sim_app.bridge", side_effect=OSError("late failure")),
            self.assertRaises(OSError),
        ):
            self.app._stop_execution("old-run")
        self.assertEqual(self.app.task["stop_confirmation"], "confirmed")
        self.assertEqual(self.app.task["stop_error"], "")

    def test_late_reply_does_not_mutate_new_task(self):
        entered, release = threading.Event(), threading.Event()

        def bridge(payload):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test timeout")
            return {"success": True}

        with patch("embodied_agent.sim_app.bridge", side_effect=bridge):
            thread = threading.Thread(target=self.app.stop)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                with self.app.lock:
                    self.app.run_id = "new-run"
                    self.app.task = {"run_id": "new-run", "phase": "planning", "events": []}
                release.set()
                thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(
                    self.app.task, {"run_id": "new-run", "phase": "planning", "events": []}
                )
            finally:
                release.set()
                thread.join(2)

    def test_original_failure_survives_cleanup_failure(self):
        with patch("embodied_agent.sim_app.bridge", side_effect=OSError("service offline")):
            self.app.run("task", "rule")
        self.assertEqual(self.app.task["phase"], "failed")
        self.assertEqual(self.app.task["failure"]["title"], "执行服务连接失败")
        self.assertEqual(self.app.task["stop_confirmation"], "unconfirmed")
        self.assertFalse(self.app.busy)

    def test_cancellation_cleanup_failure_is_not_reported_as_stopped_robot(self):
        self.app.cancel.set()
        with patch("embodied_agent.sim_app.bridge", side_effect=OSError("offline")):
            self.app.run("task", "rule")
        self.assertEqual(self.app.task["phase"], "stopped")
        self.assertIsNone(self.app.task["failure"])
        self.assertEqual(self.app.task["stop_confirmation"], "unconfirmed")
        self.assertIn("尚未确认", self.app.task["message"])


if __name__ == "__main__":
    unittest.main()
