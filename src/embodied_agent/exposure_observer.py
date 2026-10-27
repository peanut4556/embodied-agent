"""Explicit strict-capture integration for the experimental dual-clock monitor."""

from .exposure_confirmation import ExposureConfirmation
from .observation_capture import StampedPosition


class ExposureObserver:
    def __init__(self, capture, clock_map):
        if not capture.require_timestamp:
            raise ValueError("exposure observer requires strict timestamp capture")
        self.capture, self.clock_map = capture, clock_map
        self.confirmation = None
        self.next_time = None
        self.cancelled = False
        self.rows = []

    @property
    def status(self):
        if self.cancelled:
            return "cancelled"
        return self.confirmation.status if self.confirmation else "not_started"

    def step(self, task_now, events, *, stop=False):
        if stop:
            self.capture.cancel()
            if self.status in {"pending", "not_started"}:
                self.cancelled = True
                if self.confirmation:
                    self.confirmation.step(task_now, self.capture.clock(), None, stop=True)
            return self.status
        if self.status not in {"pending", "not_started"}:
            return self.status
        if self.confirmation is None:
            if not any(e["event"] == "intentional_release" for e in events):
                return self.status
            self.confirmation = ExposureConfirmation(task_now, self.clock_map)
            self.next_time = task_now
        deadline = self.confirmation.start + 3
        if task_now + 1e-9 < self.next_time and task_now < deadline - 1e-9:
            return self.status
        # Late delivery cannot trigger a new camera call after the task budget.
        point = self.capture.sample() if task_now <= deadline + 1e-9 else None
        observation = None
        if point is not None:
            observation = StampedPosition(tuple(point), self.capture.last_captured_at)
        host_now = self.capture.clock()
        self.confirmation.step(task_now, host_now, observation)
        self.rows.append(
            {
                "task_time": task_now,
                "host_time": host_now,
                "captured_at": observation.captured_at if observation else None,
                "position": point,
                "capture_error": self.capture.error,
                **self.confirmation.result(),
            }
        )
        self.next_time = task_now + 0.2
        if self.status == "pending":
            self.capture.request()
        else:
            self.capture.cancel()
        return self.status
