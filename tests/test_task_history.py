import unittest
from unittest.mock import patch

from embodied_agent.sim_app import SimulationApp


class TaskHistoryTests(unittest.TestCase):
    def setUp(self):
        self.app = SimulationApp()

    def record(self, run_id, phase="failed"):
        self.app.run_id = run_id
        self.app.task = {
            "run_id": run_id,
            "phase": phase,
            "plan": [],
            "events": [],
            "message": "result",
            "started_at": 1,
            "finished_at": 2,
            "stop_confirmation": "unconfirmed",
            "stop_error": "offline",
        }

    def test_active_and_idle_tasks_not_archived(self):
        self.assertEqual(self.app.task_history()["tasks"], [])
        self.record("active", "planning")
        del self.app.task["finished_at"]
        self.assertEqual(self.app.task_history()["tasks"], [])

    def test_bounded_newest_first_and_snapshot_isolation(self):
        for i in range(25):
            self.record(str(i))
            self.app.task_history()
        history = self.app.task_history()
        self.assertEqual(len(history["tasks"]), 20)
        self.assertEqual(history["tasks"][0]["run_id"], "24")
        self.assertEqual(history["tasks"][-1]["run_id"], "5")
        history["tasks"][0]["events"].append({"text": "modified"})
        self.assertEqual(self.app.task_history()["tasks"][0]["events"], [])

    def test_reset_keeps_record_and_new_task_does_not_overwrite_it(self):
        self.record("old")
        with patch("embodied_agent.sim_app.bridge", return_value={"success": True}):
            self.app.reset()
        self.assertEqual(self.app.task["phase"], "idle")
        with patch("embodied_agent.sim_app.threading.Thread"):
            self.app.start("把桌上的红色积木放进盒子", "rule")
        self.assertEqual(self.app.task_history()["tasks"][0]["run_id"], "old")
        self.assertNotEqual(self.app.task["run_id"], "old")

    def test_stop_retry_updates_same_record_without_duplicate(self):
        self.record("retry", "stopped")
        self.app.task_history()
        with patch("embodied_agent.sim_app.bridge", return_value={"success": True}):
            self.app.stop()
        history = self.app.task_history()["tasks"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["stop_confirmation"], "confirmed")
        self.assertEqual(history[0]["stop_error"], "")
        self.assertTrue(history[0]["events"])

    def test_failed_run_is_saved_on_completion(self):
        self.app.run_id = "failed-run"
        self.app.task.update(run_id="failed-run", started_at=1)
        with patch("embodied_agent.sim_app.bridge", side_effect=OSError("offline")):
            self.app.run("task", "rule")
        self.assertEqual(len(self.app.history), 1)
        self.assertEqual(self.app.history[0]["phase"], "failed")
        self.assertEqual(self.app.history[0]["failure"]["title"], "执行服务连接失败")
        self.assertEqual(self.app.history[0]["stop_confirmation"], "unconfirmed")


if __name__ == "__main__":
    unittest.main()
