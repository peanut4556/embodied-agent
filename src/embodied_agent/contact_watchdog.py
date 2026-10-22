"""Opt-in, single-attempt recovery from causal contact loss (development only)."""

from collections import deque

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


class DescentAcquisitionWatchdog(AcquisitionWatchdog):
    """Bounded grace for measured downward approach in this tabletop workspace."""

    def __init__(self, fps, seconds):
        super().__init__(fps, seconds)
        if seconds >= 8:
            raise ValueError("descent grace requires deadline below eight seconds")
        self.hard_deadline = 8 * fps
        self.window = int(np.ceil(0.48 * fps))
        self.tips = deque(maxlen=self.window + 1)
        self.deferred_ticks = 0
        self.last_progress = None
        self.settle_ticks = fps

    def observe(self, holding, tip_xz):
        tip = np.asarray(tip_xz, dtype=float)
        if tip.shape != (2,) or not np.isfinite(tip).all():
            raise ValueError("invalid descent observation")
        self.tips.append(tip.copy())
        elapsed = self.tick
        event = super().observe(holding)
        descending = (
            len(self.tips) == self.window + 1
            and 0.25 <= tip[0] <= 0.45
            and tip[1] >= 0.06
            and self.tips[0][1] - tip[1] >= 0.005
        )
        if elapsed >= self.deadline and descending:
            self.last_progress = elapsed
        settling = (
            self.last_progress is not None and elapsed - self.last_progress <= self.settle_ticks
        )
        if event and elapsed < self.hard_deadline and settling:
            self.finished = False
            self.deferred_ticks += 1
            return None
        return event


class GuardedExecutor:
    """Freeze learner on loss, then run the existing strict RGB-D expert once."""

    def __init__(self, world, learner, acquisition_seconds=None, descent_grace=False):
        self.world, self.learner = world, learner
        self.watchdog = ContactWatchdog(learner.fps)
        if descent_grace and acquisition_seconds is None:
            raise ValueError("descent grace requires an acquisition deadline")
        monitor = DescentAcquisitionWatchdog if descent_grace else AcquisitionWatchdog
        self.acquisition = (
            monitor(learner.fps, acquisition_seconds) if acquisition_seconds is not None else None
        )
        self.expert = None
        self.tick = 0
        self.takeover_tick = None
        self.events = []
        self.reason = ""
        self.command = world.data.ctrl.copy()

    def step(self, rgb, stop=False, *, monitor_holding=None):
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
            # Optional monitor-only observation for fault audits. Learner, expert and
            # physical scoring continue to use their original world observations.
            if monitor_holding is not None and not isinstance(monitor_holding, (bool, np.bool_)):
                raise ValueError("monitor contact must be boolean")
            contact = self.world.holding if monitor_holding is None else bool(monitor_holding)
            event = self.watchdog.observe(
                contact,
                self.world.data.site("tip").xpos[[0, 2]],
                self.world.data.ctrl,
            )
            if isinstance(self.acquisition, DescentAcquisitionWatchdog):
                acquisition_event = self.acquisition.observe(
                    contact, self.world.data.site("tip").xpos[[0, 2]]
                )
            else:
                acquisition_event = self.acquisition.observe(contact) if self.acquisition else None
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
