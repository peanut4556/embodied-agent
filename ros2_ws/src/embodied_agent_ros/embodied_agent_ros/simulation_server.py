"""ROS simulator and localhost HTTP gateway; commands complete via ROS status topics."""

import json
import os
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from embodied_agent.simulation import TabletopWorld

PREFIX = "/embodied_agent/sim/"


class Simulator(Node):
    def __init__(self):
        super().__init__("tabletop_simulator")
        if os.environ.get("SIM_ENGINE") == "mujoco":
            from embodied_agent.physics import PhysicsWorld

            self.world = PhysicsWorld(
                perception=os.environ.get("SIM_PERCEPTION", "truth"),
                policy_path=os.environ.get("SIM_POLICY_DIR"),
            )
        else:
            self.world = TabletopWorld()
        self.frame = b""
        self.frame_count = 0
        self.pending = None
        self.seen = set()
        self.cancelled_runs = set()
        self.status = self.create_publisher(String, PREFIX + "status", 10)
        self.state = self.create_publisher(String, PREFIX + "observation", 10)
        self.create_subscription(String, PREFIX + "action", self.command, 10)
        self.create_timer(0.04, self.tick)

    def finish(self, command, success, reason=""):
        self.status.publish(
            String(
                data=json.dumps(
                    {
                        "id": command["id"],
                        "success": success,
                        "reason": reason,
                        "state": self.world.snapshot(),
                    }
                )
            )
        )

    def command(self, message):
        command = json.loads(message.data)
        if command["id"] in self.seen:
            return
        if len(self.seen) > 10000:
            self.seen.clear()
        self.seen.add(command["id"])
        try:
            if command["deadline"] < time.time():
                raise ValueError("expired command")
            run_id = command.get("run_id", "")
            if command["name"] == "stop" and run_id:
                self.cancelled_runs.add(run_id)
            elif run_id and run_id in self.cancelled_runs:
                raise ValueError("task was cancelled")
            interrupted = self.pending if command["name"] == "stop" else None
            done = self.world.begin(command["name"], command.get("parameters", {}))
            if interrupted:
                self.finish(interrupted, False, "interrupted by stop")
                self.pending = None
            if command["name"] == "learned_pick_place":
                self.world.policy_motion.run_id = run_id
            if done:
                self.finish(command, True)
            else:
                self.pending = command
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            self.finish(command, False, str(exc))

    def tick(self):
        if self.pending and self.pending["deadline"] < time.time():
            self.world.begin("stop", {})
            self.finish(self.pending, False, "execution deadline exceeded")
            self.pending = None
        if self.world.tick(0.04) and self.pending:
            error = getattr(self.world, "error", "")
            self.finish(self.pending, not error, error)
            self.pending = None
        self.state.publish(String(data=json.dumps(self.world.snapshot())))
        self.frame_count += 1
        if hasattr(self.world, "render_jpeg") and self.frame_count % 5 == 0:
            self.frame = self.world.render_jpeg()


class Gateway(Node):
    def __init__(self):
        super().__init__("simulation_http_gateway")
        self.commands = queue.Queue()
        self.results = {}
        self.condition = threading.Condition()
        self.snapshot = {}
        self.publisher = self.create_publisher(String, PREFIX + "action", 10)
        self.create_subscription(String, PREFIX + "status", self.on_status, 10)
        self.create_subscription(String, PREFIX + "observation", self.on_state, 10)
        self.create_timer(0.02, self.flush)

    def on_state(self, message):
        self.snapshot = json.loads(message.data)

    def on_status(self, message):
        result = json.loads(message.data)
        with self.condition:
            if result["id"] in self.results:
                self.results[result["id"]] = result
            self.condition.notify_all()

    def flush(self):
        while not self.commands.empty():
            self.publisher.publish(String(data=json.dumps(self.commands.get_nowait())))

    def execute(self, body):
        timeout = 90 if body["name"] == "learned_pick_place" else 20
        command = {
            "id": uuid.uuid4().hex,
            "deadline": time.time() + timeout,
            "run_id": body.get("run_id", ""),
            "name": body["name"],
            "parameters": body.get("parameters", {}),
        }
        with self.condition:
            self.results[command["id"]] = None
            self.commands.put(command)
            ready = self.condition.wait_for(
                lambda: self.results[command["id"]] is not None, timeout + 2
            )
            result = self.results.pop(command["id"])
        if not ready:
            raise TimeoutError("ROS status timeout; command deadline limits motion")
        return result


def main():
    rclpy.init()
    simulator, gateway = Simulator(), Gateway()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, body):
            encoded = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path.startswith("/frame"):
                self.send_response(200 if simulator.frame else 503)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(simulator.frame)
                return
            self.respond(200, gateway.snapshot)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 16384:
                    raise ValueError("invalid request size")
                result = gateway.execute(json.loads(self.rfile.read(size)))
                self.respond(200, result)
            except (ValueError, KeyError, TypeError, TimeoutError) as exc:
                self.respond(400, {"success": False, "reason": str(exc)})

    server = ThreadingHTTPServer(("0.0.0.0", 8766), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(simulator)
    executor.add_node(gateway)
    try:
        executor.spin()
    finally:
        server.shutdown()
        executor.shutdown()
        if hasattr(simulator.world, "close"):
            simulator.world.close()
        simulator.destroy_node()
        gateway.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
