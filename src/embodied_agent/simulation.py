"""Deterministic planar arm simulation; no contact dynamics or hardware control."""

from __future__ import annotations

import math


class TabletopWorld:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tip = [0.22, 0.34]
        self.block = [0.32, 0.025]
        self.box = [0.64, 0.055]
        self.holding = False
        self.motion = []
        self.active = ""
        self.elapsed = 0.0
        self.origin = self.tip[:]

    def inside_box(self):
        return (
            not self.holding
            and abs(self.block[0] - self.box[0]) < 0.065
            and abs(self.block[1] - self.box[1]) < 0.025
        )

    def snapshot(self):
        # Two-link inverse kinematics, elbow-up solution.
        x, y = self.tip[0], self.tip[1] - 0.12
        length = 0.42
        c = max(-1.0, min(1.0, (x * x + y * y - 2 * length * length) / (2 * length * length)))
        elbow_angle = -math.acos(c)
        shoulder = math.atan2(y, x) - math.atan2(
            length * math.sin(elbow_angle), length + length * math.cos(elbow_angle)
        )
        elbow = [length * math.cos(shoulder), 0.12 + length * math.sin(shoulder)]
        return {
            "tip": self.tip[:],
            "elbow": elbow,
            "base": [0, 0.12],
            "joints": [shoulder, elbow_angle],
            "block": self.block[:],
            "box": self.box[:],
            "holding": self.holding,
            "active": self.active,
            "inside_box": self.inside_box(),
            "objects": {
                "red_block": "gripper"
                if self.holding
                else ("box" if self.inside_box() else "table"),
                "box": "table",
            },
            "gripper_holding": "red_block" if self.holding else "",
        }

    def begin(self, name, parameters):
        if name == "stop":
            self.motion = []
            self.active = ""
            return True
        if self.active:
            raise ValueError("simulator is busy")
        if name == "reset":
            self.reset()
            return True
        obj = parameters.get("object")
        if obj not in {"red_block", "box"}:
            raise ValueError("unknown object")
        if name == "locate":
            return True
        if name == "verify":
            destination = parameters.get("destination")
            if obj != "red_block" or destination not in {None, "box"} or not self.inside_box():
                raise ValueError(
                    "verification failed: block is not inside box with gripper released"
                )
            return True
        if obj != "red_block":
            raise ValueError("only red_block is graspable")
        if name == "pick":
            if self.holding:
                raise ValueError("gripper is already holding an object")
            x, y = self.block
            self.motion = [([x, 0.30], ""), ([x, y], "grasp"), ([x, 0.30], "")]
        elif name == "place":
            if not self.holding or parameters.get("destination") != "box":
                raise ValueError("place requires held red_block and destination=box")
            x, y = self.box
            self.motion = [([x, 0.30], ""), ([x, y], "release"), ([x, 0.30], "")]
        else:
            raise ValueError("unsupported simulation action")
        self.active = name
        self.elapsed = 0
        self.origin = self.tip[:]
        return False

    def tick(self, dt):
        if not self.motion:
            return False
        self.elapsed += dt
        target, event = self.motion[0]
        t = min(1.0, self.elapsed / 0.8)
        blend = t * t * (3 - 2 * t)
        self.tip = [a + (b - a) * blend for a, b in zip(self.origin, target, strict=True)]
        if self.holding:
            self.block = self.tip[:]
        if t < 1:
            return False
        if event == "grasp":
            self.holding = True
            self.block = self.tip[:]
        elif event == "release":
            self.holding = False
            self.block = self.tip[:]
        self.motion.pop(0)
        self.origin = self.tip[:]
        self.elapsed = 0
        if not self.motion:
            self.active = ""
            return True
        return False
