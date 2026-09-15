"""Evaluation-only grasp evidence; object truth is never a learned policy input."""

import numpy as np


class GraspQuality:
    def __init__(self, fps):
        if fps not in (25, 50):
            raise ValueError("unsupported quality sampling rate")
        self.fps = fps
        self.minimum = max(3, round(0.12 * fps))
        self.tick = 0
        self.contact_run = self.lift_run = self.goal_run = 0
        self.grasped = self.lifted = self.released = False
        self.events = []
        self.max_height = 0.0
        self.pending_release = False

    def observe(self, holding, xyz, command, inside):
        xyz, command = np.asarray(xyz), np.asarray(command)
        if (
            xyz.shape != (3,)
            or command.shape != (5,)
            or not np.isfinite(xyz).all()
            or not np.isfinite(command).all()
        ):
            raise ValueError("invalid quality observation")
        self.max_height = max(self.max_height, float(xyz[2]))
        self.contact_run = self.contact_run + 1 if holding else 0
        self.lift_run = self.lift_run + 1 if holding and xyz[2] >= 0.15 else 0
        self.goal_run = self.goal_run + 1 if inside and not holding else 0
        if not self.grasped and self.contact_run >= self.minimum:
            self.grasped = True
            self.events.append({"tick": self.tick, "event": "stable_bilateral_grasp"})
        if not self.lifted and self.lift_run >= self.minimum:
            self.lifted = True
            self.events.append({"tick": self.tick, "event": "sustained_lift"})
        near_tray = abs(xyz[0] - 0.64) < 0.09 and abs(xyz[1]) < 0.08
        opening = bool(np.all(command[3:] >= 0.01))
        if self.lifted and holding and near_tray and opening:
            self.pending_release = True
        if self.pending_release and not holding:
            if near_tray and opening:
                self.released = True
                self.events.append({"tick": self.tick, "event": "intentional_release_at_tray"})
            self.pending_release = False
        # A loss before intentional opening breaks the chain. A later successful
        # regrasp can establish fresh evidence; earlier lift cannot qualify a push.
        if self.lifted and not holding and not self.released and not self.pending_release:
            self.lifted = False
            self.events.append({"tick": self.tick, "event": "lift_contact_lost"})
        self.tick += 1

    def result(self):
        goal = self.goal_run >= self.fps
        qualified = bool(goal and self.grasped and self.lifted and self.released)
        return {
            "verified_pick_place": qualified,
            "stable_goal": goal,
            "stable_grasp_observed": self.grasped,
            "sustained_lift": self.lifted,
            "intentional_release": self.released,
            "max_height_m": self.max_height,
            "classification": "verified_pick_place"
            if qualified
            else ("goal_without_verified_pick_place" if goal else "task_incomplete"),
            "events": list(self.events),
        }
