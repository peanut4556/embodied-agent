"""Opt-in, single-attempt recovery from causal contact loss (development only)."""

import numpy as np

from .correction_data import CorrectionExpert


class ContactWatchdog:
    """Uses contact, arm FK and the last applied command; never object pose."""

    def __init__(self, fps):
        if fps not in (25, 50):
            raise ValueError("unsupported sampling rate")
        self.minimum = int(np.ceil(0.12 * fps))
        self.previous_holding = False
        self.lift_run = 0
        self.lifted = False
        self.finished = False

    def observe(self, holding, tip_xz, previous_command):
        tip, command = np.asarray(tip_xz), np.asarray(previous_command)
        if (
            tip.shape != (2,)
            or command.shape != (5,)
            or not np.isfinite(tip).all()
            or not np.isfinite(command).all()
        ):
            raise ValueError("invalid watchdog observation")
        if self.finished:
            return None
        self.lift_run = self.lift_run + 1 if holding and tip[1] >= 0.15 else 0
        self.lifted |= self.lift_run >= self.minimum
        event = None
        if self.previous_holding and not holding:
            intentional = (
                self.lifted
                and abs(tip[0] - 0.64) < 0.09
                and 0.02 <= tip[1] <= 0.12
                and np.all(command[3:] >= 0.01)
            )
            event = "intentional_release" if intentional else "contact_lost"
            self.finished = True
        self.previous_holding = bool(holding)
        return event


class AcquisitionWatchdog:
    """One deadline for first stable contact; no object pose or scoring signals."""

    def __init__(self, fps, seconds):
        if fps not in (25, 50) or not np.isfinite(seconds) or not 0 < seconds < 30:
            raise ValueError("invalid acquisition deadline")
        self.deadline = int(np.ceil(seconds * fps))
        self.minimum = int(np.ceil(0.12 * fps))
        self.holding_ticks = 0
        self.acquired = False
        self.finished = False
        self.tick = 0

    def observe(self, holding):
        self.holding_ticks = self.holding_ticks + 1 if holding else 0
        self.acquired |= self.holding_ticks >= self.minimum
        event = None
        # Never open the fingers during a newly forming contact grasp.
        if not self.finished and not self.acquired and not holding and self.tick >= self.deadline:
            event = "acquisition_timeout"
            self.finished = True
        self.tick += 1
        return event


class GuardedExecutor:
    """Freeze learner on loss, then run the existing strict RGB-D expert once."""

    def __init__(self, world, learner, acquisition_seconds=None):
        self.world, self.learner = world, learner
        self.watchdog = ContactWatchdog(learner.fps)
        self.acquisition = (
            AcquisitionWatchdog(learner.fps, acquisition_seconds)
            if acquisition_seconds is not None
            else None
        )
        self.expert = None
        self.tick = 0
        self.takeover_tick = None
        self.events = []
        self.reason = ""
        self.command = world.data.ctrl.copy()

    def step(self, rgb, stop=False):
        joints = self.world.data.qpos[:5]
        if self.reason:
            return self.command.copy()
        if stop or self.tick >= self.learner.max_ticks:
            self.reason = "user stop" if stop else "execution timeout"
            self.command = (
                self.expert.stop(self.reason)
                if self.expert
                else self.learner.stop(joints, self.reason)
            )
            return self.command.copy()
        if self.expert is None:
            event = self.watchdog.observe(
                self.world.holding,
                self.world.data.site("tip").xpos[[0, 2]],
                self.world.data.ctrl,
            )
            acquisition_event = (
                self.acquisition.observe(self.world.holding) if self.acquisition else None
            )
            event = event or acquisition_event
            if event:
                self.events.append({"tick": self.tick, "event": event})
            if event in {"contact_lost", "acquisition_timeout"}:
                self.takeover_tick = self.tick
                self.learner.stop(joints, f"{event} takeover")
                self.expert = CorrectionExpert(self.world, self.learner.fps, holding_ticks=0)
        self.command = (
            self.expert.step()
            if self.expert
            else self.learner.step(rgb, joints, self.world.holding)
        )
        self.reason = self.expert.reason if self.expert else self.learner.reason
        self.tick += 1
        return self.command.copy()
