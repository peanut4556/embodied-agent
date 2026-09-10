"""Run the frozen RGB policy and bounded feedback at its recorded control rate."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from .feedback import FeedbackExecutor
from .imitation import ContextPolicy


class PolicyMotion:
    def __init__(self, world, model_path):
        self.world = world
        self.renderer = None
        self.elapsed = 0.0
        self.run_id = ""
        policy = ContextPolicy(model_path)
        source = Path(__file__).parent / "assets/tabletop.xml"
        if hashlib.sha256(source.read_bytes()).hexdigest() != policy.metadata["model_sha256"]:
            raise ValueError("physics model differs from checkpoint")
        if not np.allclose(policy.bounds, world.model.actuator_ctrlrange):
            raise ValueError("checkpoint actuator bounds differ from simulator")
        if world.holding or not np.allclose(world.data.qpos[:5], policy.baseline[0], atol=0.02):
            raise ValueError("请重置场景：学习执行需要从示范的观察姿态开始")
        self.period = 1 / policy.metadata["fps"]
        self.substeps = round(self.period / world.model.opt.timestep)
        try:
            self.renderer = mujoco.Renderer(world.model, height=240, width=320)
            self.controller = FeedbackExecutor(policy, self.rgb(), world.data.qpos[:5])
            if self.controller.state == "stopped":
                raise ValueError(self.controller.reason)
        except Exception:
            self.close()
            raise

    def rgb(self):
        mujoco.mj_forward(self.world.model, self.world.data)
        self.renderer.update_scene(self.world.data, camera="perception")
        return self.renderer.render()

    def stop(self, reason="user stop"):
        self.world.data.ctrl[:] = self.controller.stop(self.world.data.qpos[:5], reason)

    def tick(self, dt, on_control_frame=None):
        count = round(dt / self.period)
        if count < 1 or not np.isclose(count * self.period, dt):
            self.stop("incompatible control clock")
        for _ in range(count):
            if self.controller.state in {"stopped", "completed"}:
                break
            if self.elapsed >= 30:
                self.stop("learned execution time limit exceeded")
                break
            try:
                rgb = self.rgb()
                holding = self.world.holding
                self.world.data.ctrl[:] = self.controller.step(
                    rgb, self.world.data.qpos[:5], holding
                )
            except (ValueError, RuntimeError) as exc:
                self.stop(f"policy execution failed: {exc}")
                break
            # Exact policy observation plus its new action, before integration.
            # Recorder errors propagate; they must not become valid failure examples.
            if on_control_frame is not None:
                on_control_frame(self.world, rgb, holding)
            try:
                for _ in range(self.substeps):
                    mujoco.mj_step(self.world.model, self.world.data)
                self.elapsed += self.period
            except (ValueError, RuntimeError) as exc:
                self.stop(f"policy execution failed: {exc}")
                break
        done = self.controller.state in {"stopped", "completed"}
        error = self.controller.reason
        if done:
            mujoco.mj_forward(self.world.model, self.world.data)
            if self.controller.state == "completed" and not self.world.inside_box():
                error = "learned execution failed: block not settled in box"
                self.stop(error)
        return done, error

    def snapshot(self):
        return {
            "mode": "feedback",
            "run_id": self.run_id,
            "state": self.controller.state,
            "reason": self.controller.reason,
            "retries": self.controller.retries,
            "slips": self.controller.slips,
            "elapsed": round(self.elapsed, 3),
            "events": list(self.controller.events),
        }

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
