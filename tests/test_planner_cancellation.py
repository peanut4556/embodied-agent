import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agent.adapters.rai_planner import RAISubprocessPlanner
from embodied_agent.models import Observation
from embodied_agent.sim_app import SimulationApp


class PlannerCancellationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.worker = self.root / "worker.py"
        self.ready = self.root / "ready"
        self.cancel = threading.Event()

    def planner(self, code, timeout=5):
        self.worker.write_text(code)
        return RAISubprocessPlanner(
            python_executable=sys.executable,
            worker_script=self.worker,
            cancel_event=self.cancel,
            timeout_seconds=timeout,
        )

    def sleeper(self, ignore=False):
        return (
            "import time, pathlib, signal\n"
            + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore else "")
            + f'pathlib.Path({str(self.ready)!r}).write_text("ready")\n'
            + "time.sleep(30)\n"
        )

    def wait_ready(self):
        deadline = time.monotonic() + 3
        while not self.ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.ready.exists())

    def test_pre_cancel_never_launches(self):
        planner = self.planner(self.sleeper())
        self.cancel.set()
        with patch("subprocess.Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                planner.create_plan("test", Observation())
            launch.assert_not_called()

    def test_normal_json_roundtrip_and_stderr(self):
        planner = self.planner(
            "import sys,json\nr=json.load(sys.stdin)\n"
            'print("diagnostic", file=sys.stderr)\n'
            'print(json.dumps({"goal": r["instruction"], "steps": []}))\n'
        )
        self.assertEqual(planner.create_plan("中文任务", Observation()).goal, "中文任务")

    def test_failure_and_invalid_json(self):
        for code, message in [
            ('import sys;sys.stderr.write("bad worker");sys.exit(2)', "bad worker"),
            ('print("bad json")', "invalid JSON"),
        ]:
            with self.subTest(code=code), self.assertRaisesRegex(RuntimeError, message):
                self.planner(code).create_plan("test", Observation())

    def check_running_cancel(self, ignore):
        import subprocess

        planner = self.planner(self.sleeper(ignore))
        errors = []

        def execute():
            try:
                planner.create_plan("test", Observation())
            except RuntimeError as error:
                errors.append(str(error))

        original = subprocess.Popen
        processes = []

        def launch(*args, **kwargs):
            process = original(*args, **kwargs)
            processes.append(process)
            return process

        with patch("subprocess.Popen", side_effect=launch):
            thread = threading.Thread(target=execute)
            thread.start()
            try:
                self.wait_ready()
                start = time.monotonic()
                self.cancel.set()
                thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertLess(time.monotonic() - start, 2)
                self.assertEqual(errors, ["RAI planning cancelled"])
                self.assertIsNotNone(processes[0].poll())
                self.assertTrue(
                    all(
                        stream.closed
                        for stream in (processes[0].stdin, processes[0].stdout, processes[0].stderr)
                    )
                )
            finally:
                self.cancel.set()
                thread.join(3)

    def test_running_cancel_reaps_worker(self):
        self.check_running_cancel(False)

    @unittest.skipIf(os.name == "nt", "POSIX TERM handler")
    def test_unresponsive_worker_is_killed(self):
        self.check_running_cancel(True)

    def test_timeout_terminates_worker(self):
        planner = self.planner(self.sleeper(), timeout=0.15)
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            planner.create_plan("test", Observation())
        self.assertLess(time.monotonic() - start, 2)

    def test_console_stop_releases_busy_without_motion(self):
        app = SimulationApp()
        planner = self.planner(self.sleeper())
        commands = []

        def bridge(payload=None):
            if payload is None:
                return {"active": "", "objects": {}, "gripper_holding": ""}
            commands.append(payload["name"])
            return {"success": True}

        def backend(**kwargs):
            planner.cancel_event = kwargs["cancel_event"]
            return planner

        with (
            patch("embodied_agent.sim_app.bridge", side_effect=bridge),
            patch("embodied_agent.sim_app.RAISubprocessPlanner", side_effect=backend),
        ):
            app.start("test", "rai")
            try:
                self.wait_ready()
                start = time.monotonic()
                app.stop()
                while app.busy and time.monotonic() - start < 2:
                    time.sleep(0.01)
                self.assertFalse(app.busy)
                self.assertEqual(app.task["phase"], "stopped")
                self.assertEqual(app.task["completed"], 0)
                self.assertTrue(all(command == "stop" for command in commands))
                app.reset()
                self.assertEqual(app.task["phase"], "idle")
            finally:
                app.cancel.set()


if __name__ == "__main__":
    unittest.main()
