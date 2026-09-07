"""MuJoCo contact simulation. Cube qpos is never set during action execution."""

from __future__ import annotations

import io
import math
from pathlib import Path

import mujoco
import numpy as np


class PhysicsWorld:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(
            str(Path(__file__).parent / "assets/tabletop.xml")
        )
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.reset()

    @staticmethod
    def ik(x, z):
        z += 0.075 - 0.12
        c = (x * x + z * z - 2 * 0.42**2) / (2 * 0.42**2)
        if not -1 <= c <= 1:
            raise ValueError("target outside arm workspace")
        elbow = -math.acos(c)
        shoulder = math.atan2(z, x) - math.atan2(math.sin(elbow), 1 + math.cos(elbow))
        return np.array([-shoulder, -elbow, shoulder + elbow])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.target = np.array([0.22, 0.30])
        self.data.qpos[:3] = self.ik(*self.target)
        self.data.qpos[3:5] = 0.04
        self.data.ctrl[:3] = self.data.qpos[:3]
        self.data.ctrl[3:5] = 0.04
        self.active = ""
        self.motion = []
        self.elapsed = 0.0
        self.origin = self.target.copy()
        self.error = ""
        mujoco.mj_forward(self.model, self.data)
        for _ in range(150):
            mujoco.mj_step(self.model, self.data)

    def contacts(self):
        cube = self.model.geom("red_block").id
        fingers = {self.model.geom(name).id for name in ("finger_left", "finger_right")}
        touching = set()
        for contact in self.data.contact:
            pair = {contact.geom1, contact.geom2}
            if cube in pair:
                touching |= pair & fingers
        return len(touching)

    @property
    def holding(self):
        return bool(self.contacts() == 2 and self.data.ctrl[3] < 0.01)

    def inside_box(self):
        block = self.data.body("red_block").xpos
        speed = np.linalg.norm(self.data.qvel[5:8])
        return bool(
            abs(block[0] - 0.64) < 0.06
            and abs(block[1]) < 0.05
            and 0.030 < block[2] < 0.06
            and not self.holding
            and speed < 0.03
        )

    def snapshot(self):
        block = self.data.body("red_block").xpos.copy()
        tip = self.data.site("tip").xpos.copy()
        elbow = self.data.body("forearm").xpos
        inside, holding = self.inside_box(), self.holding
        return {
            "engine": "mujoco",
            "sim_time": float(self.data.time),
            "tip": tip[[0, 2]].tolist(),
            "elbow": elbow[[0, 2]].tolist(),
            "base": [0, 0.12],
            "joints": (-self.data.qpos[:2]).tolist(),
            "block": block[[0, 2]].tolist(),
            "block_xyz": block.tolist(),
            "box": [0.64, 0.04],
            "holding": holding,
            "active": self.active,
            "inside_box": inside,
            "finger_contacts": self.contacts(),
            "error": self.error,
            "objects": {
                "red_block": "gripper" if holding else ("box" if inside else "table"),
                "box": "table",
            },
            "gripper_holding": "red_block" if holding else "",
        }

    def begin(self, name, parameters):
        if name == "stop":
            self.motion = []
            self.active = ""
            # Hold current joint angles with actuators. Gravity/contact continue to run.
            self.data.ctrl[:3] = self.data.qpos[:3]
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
            if (
                obj != "red_block"
                or parameters.get("destination") not in {None, "box"}
                or not self.inside_box()
            ):
                raise ValueError(
                    "verification failed: block not settled inside box with gripper released"
                )
            return True
        if obj != "red_block":
            raise ValueError("only red_block is graspable")
        block = self.data.body("red_block").xpos.copy()
        if name == "pick":
            if self.holding or abs(block[1]) > 0.015 or block[2] < 0.02:
                raise ValueError("block not graspable from current pose")
            x, z = float(block[0]), float(block[2])
            self.motion = [
                ([x, 0.30], 0.04, 1.2),
                ([x, z + 0.002], 0.04, 1.2),
                ([x, z + 0.002], 0.0, 0.8),
                ([x, 0.30], 0.0, 1.5),
            ]
        elif name == "place":
            if not self.holding or parameters.get("destination") != "box":
                raise ValueError("place requires contact grasp and destination=box")
            self.motion = [
                ([0.64, 0.30], 0.0, 1.5),
                ([0.64, 0.048], 0.0, 1.2),
                ([0.64, 0.048], 0.04, 0.8),
                ([0.64, 0.30], 0.04, 1.2),
            ]
        else:
            raise ValueError("unsupported physics action")
        self.origin = self.data.site("tip").xpos[[0, 2]].copy()
        self.elapsed = 0.0
        self.active = name
        self.error = ""
        return False

    def tick(self, dt):
        finished = False
        for _ in range(round(dt / self.model.opt.timestep)):
            if self.motion:
                target, grip, duration = self.motion[0]
                self.elapsed += self.model.opt.timestep
                t = min(1, self.elapsed / duration)
                blend = t * t * (3 - 2 * t)
                self.target = self.origin + (np.array(target) - self.origin) * blend
                self.data.ctrl[:3] = self.ik(*self.target)
                self.data.ctrl[3:5] = grip
                if t >= 1:
                    self.motion.pop(0)
                    self.origin = self.target.copy()
                    self.elapsed = 0
                    if not self.motion:
                        action = self.active
                        self.active = ""
                        if action == "pick" and (
                            not self.holding or self.data.body("red_block").xpos[2] < 0.15
                        ):
                            self.error = (
                                "grasp failed: no stable bilateral contact or cube not lifted"
                            )
                        if action == "place" and not self.inside_box():
                            self.error = "place failed: cube did not settle inside tray"
                        finished = True
            mujoco.mj_step(self.model, self.data)
        return finished

    def render_jpeg(self):
        from PIL import Image

        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=720)
        self.renderer.update_scene(self.data, camera="overview")
        buffer = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()
