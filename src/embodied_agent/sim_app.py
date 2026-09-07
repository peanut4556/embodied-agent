"""Local simulation console: RAI on macOS, ROS execution in Docker."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from .adapters.rai_planner import RAISubprocessPlanner
from .models import Observation
from .planner import RuleBasedPlanner
from .safety import SafetyGate

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = "http://127.0.0.1:8766"


def bridge(payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(BRIDGE, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=25) as response:
        result = json.load(response)
    if payload is not None and not result.get("success"):
        raise RuntimeError(result.get("reason", "ROS command failed"))
    return result


class SimulationApp:
    def __init__(self):
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.busy = False
        self.run_id = ""
        self.task = {"phase": "idle", "plan": [], "events": [], "message": "准备就绪"}

    def update(self, **values):
        with self.lock:
            self.task.update(values)

    def event(self, text):
        with self.lock:
            self.task["events"].append({"time": time.strftime("%H:%M:%S"), "text": text})

    def start(self, instruction, planner):
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1000:
            raise ValueError("请输入 1–1000 字的任务")
        if planner not in {"rai", "rule"}:
            raise ValueError("unknown planner")
        if planner == "rule" and instruction != "把桌上的红色积木放进盒子":
            raise ValueError("规则演示仅支持默认任务，请选择 Qwen 处理其他指令")
        with self.lock:
            if self.busy:
                raise ValueError("任务仍在运行")
            self.busy = True
            self.cancel.clear()
            self.run_id = uuid.uuid4().hex
            self.task = {
                "phase": "planning",
                "plan": [],
                "events": [],
                "message": "Qwen 正在规划" if planner == "rai" else "生成规则计划",
            }
        threading.Thread(target=self.run, args=(instruction, planner), daemon=True).start()

    def run(self, instruction, planner):
        try:
            state = bridge()
            if not state or state["active"]:
                raise RuntimeError("仿真未就绪或仍在执行")
            observation = Observation(
                objects=state["objects"], gripper_holding=state["gripper_holding"]
            )
            backend = RAISubprocessPlanner() if planner == "rai" else RuleBasedPlanner()
            self.event("发送场景观测给规划器")
            plan = backend.create_plan(instruction, observation)
            SafetyGate().validate_plan(plan)
            self.update(plan=[asdict(step) for step in plan.steps])
            self.event(f"计划通过校验，共 {len(plan.steps)} 步")
            stopped = False
            for index, step in enumerate(plan.steps):
                with self.lock:
                    if self.cancel.is_set():
                        raise RuntimeError("任务已由用户停止")
                    self.task.update(
                        phase="executing", current=index, message=f"执行 {step.action}"
                    )
                if step.action == "stop":
                    stopped = True
                    break
                self.event(f"ROS → {step.action} {json.dumps(step.arguments, ensure_ascii=False)}")
                result = bridge(
                    {"name": step.action, "parameters": step.arguments, "run_id": self.run_id}
                )
                if self.cancel.is_set():
                    raise RuntimeError("任务已由用户停止")
                self.event(f"ROS ← 完成 {step.action} · {result['id'][:8]}")
            final = bridge()
            if self.cancel.is_set():
                raise RuntimeError("任务已由用户停止")
            if not final["inside_box"] or final["holding"]:
                if stopped:
                    raise RuntimeError("规划器要求停止；未完成放置任务")
                raise RuntimeError("位置验收失败：积木未进入盒子或夹爪尚未松开")
            self.event("位置验收通过：积木在盒内，夹爪已松开")
            self.update(phase="success", message="完成：红色积木已进入盒子")
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            self.event(str(exc))
            self.update(phase="stopped" if self.cancel.is_set() else "failed", message=str(exc))
            try:
                bridge({"name": "stop", "run_id": self.run_id})
            except (OSError, RuntimeError, ValueError) as stop_error:
                self.event(f"停止反馈不可用：{stop_error}")
        finally:
            with self.lock:
                self.busy = False

    def stop(self):
        self.cancel.set()
        bridge({"name": "stop", "run_id": self.run_id})
        self.event("用户发出停止指令")

    def reset(self):
        with self.lock:
            if self.busy:
                raise ValueError("请先停止任务并等待规划或执行结束")
            bridge({"name": "reset"})
            self.task = {"phase": "idle", "plan": [], "events": [], "message": "场景已重置"}


def main():
    app = SimulationApp()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, payload):
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode())

        def do_GET(self):
            if self.path.startswith("/api/frame"):
                try:
                    with urlopen(BRIDGE + "/frame", timeout=3) as response:
                        image = response.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(image)
                except OSError as exc:
                    self.respond(503, {"error": str(exc)})
            elif self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write((ROOT / "web/simulation.html").read_bytes())
            elif self.path == "/api/state":
                try:
                    world = bridge()
                    with app.lock:
                        task = json.loads(json.dumps(app.task))
                        busy = app.busy
                    self.respond(200, {"world": world, "task": task, "busy": busy})
                except (OSError, RuntimeError, ValueError) as exc:
                    self.respond(503, {"error": str(exc)})
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self):
            # JSON-only local API prevents browser form / cross-origin submissions.
            if self.headers.get("Content-Type") != "application/json":
                self.respond(415, {"error": "application/json required"})
                return
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://localhost:8765", "http://127.0.0.1:8765"}:
                self.respond(403, {"error": "local origin required"})
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size < 16384:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(size))
                if self.path == "/api/run":
                    app.start(payload.get("instruction"), payload.get("planner", "rai"))
                elif self.path == "/api/stop":
                    app.stop()
                elif self.path == "/api/reset":
                    app.reset()
                else:
                    raise ValueError("unknown action")
                self.respond(200, {"ok": True})
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                self.respond(400, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Simulation console: http://127.0.0.1:8765", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
