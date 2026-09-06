from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .protocol import status_json, validate_action_json


class ActionBridge(Node):
    """Validate semantic Agent actions before a hardware driver sees them."""

    def __init__(self) -> None:
        super().__init__("embodied_agent_action_bridge")
        self._status_publisher = self.create_publisher(String, "/embodied_agent/status", 10)
        self._action_subscription = self.create_subscription(
            String, "/embodied_agent/action", self._on_action, 10
        )
        self.get_logger().info("action bridge ready")

    def _on_action(self, message: String) -> None:
        response = String()
        try:
            action = validate_action_json(message.data)
        except (TypeError, ValueError) as error:
            response.data = status_json(accepted=False, reason=str(error))
            self.get_logger().warning(str(error))
        else:
            # A real robot adapter will forward this only after joint/workspace checks.
            response.data = status_json(accepted=True, action=action["name"])
            self.get_logger().info(f"accepted action={action['name']}")
        self._status_publisher.publish(response)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ActionBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
