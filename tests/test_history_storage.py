import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agent.sim_app import SimulationApp
from embodied_agent.task_history import TaskHistoryStore


def record(run_id="one"):
    return {
        "run_id": run_id,
        "phase": "failed",
        "started_at": 1,
        "finished_at": 2,
        "plan": [],
        "events": [],
        "message": "failed",
        "stop_confirmation": "unconfirmed",
    }


class HistoryStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "history.json"

    def test_restart_preserves_history_but_not_active_state(self):
        app = SimulationApp(history_path=self.path)
        app.task = record()
        app.task_history()
        restarted = SimulationApp(history_path=self.path)
        self.assertEqual(restarted.task_history()["tasks"], [record()])
        self.assertEqual(restarted.task["phase"], "idle")
        self.assertFalse(restarted.busy)
        self.assertEqual(restarted.task_history()["scope"], "local")

    def test_fresh_python_process_loads_saved_records(self):
        TaskHistoryStore(self.path).save([record()])
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys; from embodied_agent.sim_app import SimulationApp; "
                    "print(json.dumps(SimulationApp(history_path=sys.argv[1]).task_history()))"
                ),
                str(self.path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["tasks"], [record()])

    def test_unchanged_history_does_not_write_again(self):
        store = TaskHistoryStore(self.path)
        store.save([record()])
        with patch("embodied_agent.task_history.os.replace") as replace:
            store.save([record()])
            replace.assert_not_called()

    def test_replace_failure_preserves_old_file_and_retry_clears_warning(self):
        store = TaskHistoryStore(self.path)
        store.save([record()])
        old = self.path.read_bytes()
        with patch("embodied_agent.task_history.os.replace", side_effect=OSError("disk error")):
            store.save([record("two")])
        self.assertTrue(store.warning)
        self.assertEqual(self.path.read_bytes(), old)
        self.assertEqual(list(self.path.parent.glob(".task-history-*")), [])
        store.save([record("two")])
        self.assertEqual(store.warning, "")
        self.assertEqual(TaskHistoryStore(self.path).load()[0]["run_id"], "two")

    def test_corrupt_or_unknown_version_is_not_overwritten(self):
        for content in (
            "{broken",
            '{"version":2,"tasks":[]}',
            json.dumps({"version": 1, "tasks": [record(), record()]}),
        ):
            with self.subTest(content=content):
                self.path.write_text(content)
                store = TaskHistoryStore(self.path)
                self.assertEqual(store.load(), [])
                store.save([record("new")])
                self.assertTrue(store.blocked)
                self.assertEqual(self.path.read_text(), content)
                self.assertTrue(store.warning)

    def test_read_error_is_nonfatal_and_preserves_source(self):
        self.path.mkdir()
        store = TaskHistoryStore(self.path)
        self.assertEqual(store.load(), [])
        self.assertTrue(store.warning)
        self.assertTrue(self.path.is_dir())

    def test_capacity_persists_newest_twenty(self):
        app = SimulationApp(history_path=self.path)
        for i in range(25):
            app.task = record(str(i))
            app.task_history()
        tasks = SimulationApp(history_path=self.path).task_history()["tasks"]
        self.assertEqual(len(tasks), 20)
        self.assertEqual([tasks[0]["run_id"], tasks[-1]["run_id"]], ["24", "5"])

    def test_invalid_active_record_and_oversize_file_rejected(self):
        store = TaskHistoryStore(self.path)
        active = record()
        active["phase"] = "executing"
        store.save([active])
        self.assertTrue(store.warning)
        self.assertFalse(self.path.exists())
        self.path.write_text(" " * 2_000_001)
        self.assertEqual(store.load(), [])
        self.assertTrue(store.blocked)

    def test_stop_confirmation_update_survives_restart(self):
        app = SimulationApp(history_path=self.path)
        app.run_id = "one"
        app.task = record()
        app.task["phase"] = "stopped"
        app.task_history()
        with patch("embodied_agent.sim_app.bridge", return_value={"success": True}):
            app.stop()
        task = SimulationApp(history_path=self.path).task_history()["tasks"][0]
        self.assertEqual(task["stop_confirmation"], "confirmed")
        self.assertTrue(task["events"])


if __name__ == "__main__":
    unittest.main()
