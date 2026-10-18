"""Fault proxy between a real console and live ROS; backend is never restarted."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from check_console_interactions import TASK, wait_until
from check_simulation import request, wait_task

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("outputs/stop-disconnect-v1.json"))
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite report")
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", 8765)) != 0, "existing console must be stopped first"
    initial = request(8766)
    assert initial["engine"] == "mujoco" and initial["perception"] == "rgbd"
    assert not initial["active"], "do not interrupt an existing motion"
    blocked = threading.Event()

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def forward(self):
            if blocked.is_set():
                self.send_error(503, "Injected test disconnect")
                return
            data = (
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.command == "POST"
                else None
            )
            try:
                with urlopen(
                    Request(
                        "http://127.0.0.1:8766" + self.path,
                        data=data,
                        headers={"Content-Type": "application/json"},
                    ),
                    timeout=25,
                ) as r:
                    body = r.read()
                    self.send_response(r.status)
                    self.send_header(
                        "Content-Type", r.headers.get("Content-Type", "application/json")
                    )
                    self.end_headers()
                    self.wfile.write(body)
            except (OSError, URLError) as exc:
                self.send_error(502, str(exc))

        do_GET = forward
        do_POST = forward

    server = ThreadingHTTPServer(("127.0.0.1", 8767), Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    report = {
        "status": "running",
        "fault": "HTTP 503 for new requests at local proxy; already forwarded action stays in flight",
        "checks": {},
    }
    process = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                'import embodied_agent.sim_app as app; app.BRIDGE="http://127.0.0.1:8767"; app.main()',
            ],
            cwd=ROOT,
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            try:
                request(8765, "api/state")
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError("test console exited")
                time.sleep(0.05)
        else:
            raise TimeoutError("console startup timeout")
        request(8765, "api/reset", {})
        request(8765, "api/run", TASK)
        active = wait_until(lambda s: s["world"]["active"] == "pick")
        report["active_before_fault"] = active
        blocked.set()
        try:
            request(8765, "api/stop", {})
        except HTTPError as error:
            assert error.code == 400
            report["stop_http_error"] = json.load(error)
        else:
            raise AssertionError("stop unexpectedly acknowledged across broken link")
        offline = wait_task(timeout=35)
        report["offline"] = offline
        assert offline["world"] is None and offline["connection_error"]
        assert offline["task"]["phase"] == "stopped"
        assert offline["task"]["stop_confirmation"] == "unconfirmed"
        assert offline["task"]["stop_error"] and "尚未确认" in offline["task"]["message"]
        report["backend_during_fault"] = request(8766)
        report["checks"]["offline_cancel_is_not_stop_ack"] = True
        blocked.clear()
        request(8765, "api/stop", {})
        recovered = request(8765, "api/state")
        report["recovered"] = recovered
        assert recovered["task"]["run_id"] == active["task"]["run_id"]
        assert recovered["task"]["stop_confirmation"] == "confirmed"
        assert recovered["task"]["stop_error"] == "" and not recovered["world"]["active"]
        assert recovered["world"]["sim_time"] >= report["backend_during_fault"]["sim_time"]
        report["checks"]["retry_confirms_same_run_without_backend_reset"] = True
        request(8765, "api/reset", {})
        request(8765, "api/run", TASK)
        restarted = wait_task()
        report["restarted"] = restarted
        assert restarted["task"]["phase"] == "success"
        assert restarted["task"]["stop_confirmation"] == "not_requested"
        assert restarted["world"]["inside_box"] and not restarted["world"]["holding"]
        report["checks"]["new_run_clears_confirmation_and_completes"] = True
        request(8765, "api/reset", {})
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        blocked.clear()
        if process is not None:
            try:
                request(8765, "api/stop", {})
            except OSError as exc:
                report["cleanup_error"] = str(exc)
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        server.shutdown()
        server.server_close()
        report["source_sha256"] = {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "scripts/check_stop_disconnect.py",
                "src/embodied_agent/sim_app.py",
                "ros2_ws/src/embodied_agent_ros/embodied_agent_ros/simulation_server.py",
            )
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "checks": report["checks"]}), flush=True)


if __name__ == "__main__":
    main()
