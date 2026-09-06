from __future__ import annotations

import json
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from .action_bridge import ActionBridge


class LoopbackProbe(Node):
    def __init__(self) -> None:
        super().__init__("embodied_agent_loopback_probe")
        self.status: dict[str, object] | None = None
        self._publisher = self.create_publisher(String, "/embodied_agent/action", 10)
        self._subscription = self.create_subscription(
            String, "/embodied_agent/status", self._on_status, 10
        )

    def publish_demo_action(self) -> None:
        message = String()
        message.data = json.dumps({"name": "pick", "parameters": {"object": "red_block"}})
        self._publisher.publish(message)

    def _on_status(self, message: String) -> None:
        self.status = json.loads(message.data)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    bridge = ActionBridge()
    probe = LoopbackProbe()
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    executor.add_node(probe)

    try:
        deadline = time.monotonic() + 5.0
        sent = False
        while probe.status is None and time.monotonic() < deadline:
            if not sent:
                probe.publish_demo_action()
                sent = True
            executor.spin_once(timeout_sec=0.1)

        if probe.status is None:
            raise RuntimeError("timed out waiting for /embodied_agent/status")
        if probe.status.get("accepted") is not True:
            raise RuntimeError(f"bridge rejected demo action: {probe.status}")
        print(json.dumps(probe.status, ensure_ascii=False, sort_keys=True))
    finally:
        executor.remove_node(probe)
        executor.remove_node(bridge)
        probe.destroy_node()
        bridge.destroy_node()
        rclpy.shutdown()
