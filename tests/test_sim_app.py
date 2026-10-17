import threading
import time
import unittest
from unittest.mock import patch

from embodied_agent.sim_app import SimulationApp, failure_details


class SimulationAppTests(unittest.TestCase):
    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        self.fail("task did not reach expected state")

    def bridge(self, payload=None):
        if payload is None:
            return {
                "active": "",
                "objects": {"red_block": "table", "box": "table"},
                "gripper_holding": "",
                "inside_box": True,
                "holding": False,
            }
        self.commands.append(payload["name"])
        if payload["name"] == "locate":
            self.entered.set()
            if not self.release.wait(2):
                raise RuntimeError("test release timeout")
        return {"success": True, "id": "test-command", "state": {}}

    def setUp(self):
        self.app = SimulationApp()
        self.commands = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def tearDown(self):
        self.release.set()
        with self.app.condition:
            self.app.cancel.set()
            self.app.condition.notify_all()
        self.wait_for(lambda: not self.app.busy)

    @patch("embodied_agent.sim_app.bridge")
    def test_pause_boundary_resume_and_progress(self, bridge):
        bridge.side_effect = self.bridge
        self.app.start("把桌上的红色积木放进盒子", "rule")
        self.assertTrue(self.entered.wait(2))
        self.app.pause()
        self.assertEqual(self.app.task["phase"], "pause_pending")
        self.release.set()
        self.wait_for(lambda: self.app.task["phase"] == "paused")
        self.assertEqual(self.commands, ["locate"])
        self.assertEqual(self.app.task["completed"], 1)
        self.app.pause()  # repeated pause is idempotent
        self.assertEqual(self.app.task["phase"], "paused")
        with self.assertRaises(ValueError):
            self.app.start("把桌上的红色积木放进盒子", "rule")
        with self.assertRaises(ValueError):
            self.app.reset()
        self.app.resume()
        self.wait_for(lambda: not self.app.busy)
        self.assertEqual(self.app.task["phase"], "success")
        self.assertEqual(self.app.task["completed"], len(self.app.task["plan"]))
        self.assertGreaterEqual(self.app.task["finished_at"], self.app.task["started_at"])

    @patch("embodied_agent.sim_app.bridge")
    def test_stop_wakes_paused_worker_without_next_action(self, bridge):
        bridge.side_effect = self.bridge
        self.app.start("把桌上的红色积木放进盒子", "rule")
        self.assertTrue(self.entered.wait(2))
        self.app.pause()
        self.release.set()
        self.wait_for(lambda: self.app.task["phase"] == "paused")
        self.app.stop()
        self.wait_for(lambda: not self.app.busy)
        self.assertEqual(self.app.task["phase"], "stopped")
        self.assertNotIn("pick", self.commands)
        self.assertIsNone(self.app.task["failure"])

    @patch("embodied_agent.sim_app.bridge")
    def test_failure_has_actionable_detail_and_stops(self, bridge):
        def fail(payload=None):
            if payload is None:
                raise OSError("connection refused")
            return {"success": True}

        bridge.side_effect = fail
        self.app.start("把桌上的红色积木放进盒子", "rule")
        self.wait_for(lambda: not self.app.busy)
        self.assertEqual(self.app.task["phase"], "failed")
        self.assertEqual(self.app.task["failure"]["title"], "执行服务连接失败")
        self.assertIn("connection refused", self.app.task["failure"]["detail"])
        self.assertTrue(self.app.task["failure"]["recovery"])
        bridge.assert_called_with({"name": "stop", "run_id": self.app.run_id})

    def test_idle_pause_resume_rejected_and_input_validated(self):
        for action in (self.app.pause, self.app.resume):
            with self.assertRaises(ValueError):
                action()
        for instruction in ("", " ", "x" * 1001):
            with self.assertRaises(ValueError):
                self.app.start(instruction, "rule")

    @patch("embodied_agent.sim_app.RAISubprocessPlanner")
    @patch("embodied_agent.sim_app.bridge")
    def test_stop_during_planning_discards_late_plan(self, bridge, planner):
        from embodied_agent.planner import RuleBasedPlanner

        bridge.side_effect = self.bridge

        def delayed_plan(instruction, observation):
            self.entered.set()
            if not self.release.wait(2):
                raise RuntimeError("test release timeout")
            return RuleBasedPlanner().create_plan(instruction, observation)

        planner.return_value.create_plan.side_effect = delayed_plan
        self.app.start("把桌上的红色积木放进盒子", "rai", "scripted")
        self.assertTrue(self.entered.wait(2))
        self.app.stop()
        self.assertTrue(self.app.busy)
        self.release.set()
        self.wait_for(lambda: not self.app.busy)
        self.assertEqual(self.app.task["phase"], "stopped")
        self.assertEqual(self.app.task["completed"], 0)
        self.assertTrue(self.commands)
        self.assertTrue(all(command == "stop" for command in self.commands))

    def test_failure_categories(self):
        for message, title in [
            ("block not graspable", "未能可靠定位或抓住积木"),
            ("execution deadline exceeded", "规划或执行超时"),
            ("学习模型未配置", "学习模型不可用"),
        ]:
            self.assertEqual(failure_details(RuntimeError(message))["title"], title)


if __name__ == "__main__":
    unittest.main()
