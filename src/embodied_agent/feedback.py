"""Bounded feedback supervisor around the frozen visual-context learned policy.

Inputs are RGB, joint encoders and simulated bilateral gripper contact. Object
truth poses belong exclusively to the separate evaluation fixture.
"""

from __future__ import annotations

import numpy as np

from .imitation import visual_context


class FeedbackExecutor:
    def __init__(self, policy, initial_rgb, initial_joints, feedback=True, max_retries=2):
        self.policy = policy
        self.fps = policy.metadata["fps"]
        self.feedback = feedback
        self.max_retries = max_retries
        if type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("max_retries must be between 0 and 3")
        self.home = np.array(initial_joints, dtype=float).copy()
        if self.home.shape != (5,) or not np.isfinite(self.home).all():
            raise ValueError("invalid initial joint observation")
        self.command = np.clip(self.home, policy.bounds[:, 0], policy.bounds[:, 1])
        self.rate_limit = np.array([2.0, 2.0, 2.0, 1.0, 1.0]) / self.fps
        self.state, self.reason = "running", ""
        self.index, self.ticks, self.retries = 0, 0, 0
        self.return_ticks, self.stable_ticks = 0, 0
        self.events = []
        self.actions = None
        self.context = None
        try:
            self._replan(initial_rgb)
        except ValueError as exc:
            self.stop(self.home, str(exc))

    def _replan(self, rgb):
        self.actions = self.policy.plan(rgb)
        self.context = visual_context(rgb)
        self.index = 0
        self.state = "running"
        self.events.append({"tick": self.ticks, "event": "planned", "context": self.context})

    def stop(self, joints, reason="user stop"):
        # Preserve the existing finger target so a stop does not intentionally release a grasp.
        if self.state != "stopped":
            self.command[:3] = np.clip(
                joints[:3], self.policy.bounds[:3, 0], self.policy.bounds[:3, 1]
            )
            self.state, self.reason = "stopped", reason
            self.actions = None
            self.events.append({"tick": self.ticks, "event": "stopped", "reason": reason})
        return self.command.copy()

    def _limit(self, target):
        target = np.clip(target, self.policy.bounds[:, 0], self.policy.bounds[:, 1])
        self.command += np.clip(target - self.command, -self.rate_limit, self.rate_limit)
        return self.command.copy()

    def _recover(self, joints, reason):
        if self.retries >= self.max_retries:
            return self.stop(joints, "retry limit reached")
        self.retries += 1
        self.state, self.return_ticks, self.stable_ticks = "returning", 0, 0
        self.actions = None
        self.events.append(
            {"tick": self.ticks, "event": "recovering", "reason": reason, "attempt": self.retries}
        )
        return self._limit(self.home)

    def step(self, rgb, joints, holding, stop=False):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (5,) or not np.isfinite(joints).all():
            return self.stop(self.command, "invalid joint observation")
        self.ticks += 1
        if stop:
            return self.stop(joints)
        if self.state in {"stopped", "completed"}:
            return self.command.copy()
        if self.state == "returning":
            self.return_ticks += 1
            settled = np.all(np.abs(joints - self.home) < [0.01, 0.01, 0.01, 0.003, 0.003])
            self.stable_ticks = self.stable_ticks + 1 if settled else 0
            if self.return_ticks > 3 * self.fps:
                return self.stop(joints, "return-to-observation timeout")
            if self.stable_ticks >= 3:
                try:
                    self._replan(rgb)
                except ValueError as exc:
                    return self.stop(joints, f"relocalization failed: {exc}")
            return self._limit(self.home)
        if self.index >= len(self.actions):
            self.state = "completed"
            return self.command.copy()
        # The demonstrated lift ends at ~4.72 s; contact must exist before transferring.
        if self.feedback and self.index == round(4.72 * self.fps) and not holding:
            return self._recover(joints, "grasp contact absent after lift")
        # During early approach, a reliable visible displacement can trigger an earlier retry.
        if (
            self.feedback
            and not holding
            and 0 < self.index <= round(0.6 * self.fps)
            and self.index % max(1, self.fps // 5) == 0
        ):
            try:
                context = visual_context(rgb)
            except ValueError:
                # The moving gripper can occlude the cube. The later contact gate is mandatory.
                context = None
            if context is not None and abs(context - self.context) > 2 / 320:
                return self._recover(joints, "visible target displacement")
        action = self.actions[self.index]
        self.index += 1
        return self._limit(action)
