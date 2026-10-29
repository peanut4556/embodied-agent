"""Explicit process recovery for one release window; no automatic calibration."""

import math
import time

from .exposure_confirmation import ExposureClockMap, ExposureConfirmation


class ProcessExposureObserver:
    """Single owner; caller supplies release time and explicitly calibrated mappings.

    restart/close perform process cleanup and belong outside the servo loop.
    The original task deadline survives every restart and mapping replacement.
    """

    def __init__(self, session, start_time, *, clock=time.monotonic):
        if not math.isfinite(start_time):
            raise ValueError("finite release time required")
        self.session, self.start, self.clock = session, start_time, clock
        self.session_id = session.session
        self.confirmation = None
        self.last_task = None
        self.status, self.reason = "pending", "awaiting_clock_mapping"
        self.rows = []

    def _invalidate(self, reason):
        if self.confirmation is None and self.reason == reason:
            return
        self.confirmation = None
        self.reason = reason
        self.rows.append({"event": reason, "session": self.session.session})

    def bind_mapping(self, session_id, mapping, task_now):
        """Bind only an idle, current session to a newly measured clock origin."""
        if self.status != "pending":
            raise RuntimeError("release window is terminal")
        if (
            session_id != self.session.session
            or self.session.status != "idle"
            or not isinstance(mapping, ExposureClockMap)
            or mapping.host_origin < self.session.started
        ):
            raise ValueError("fresh mapping for the current idle camera session required")
        self._validate_time(task_now)
        if task_now >= self.start + 3:
            raise ValueError("release deadline reached")
        now = self.clock()
        mapping.map(mapping.host_origin, now, task_now)
        self._invalidate("clock_mapping_bound")
        self.session_id = session_id
        self.confirmation = ExposureConfirmation(self.start, mapping)
        self.reason = "collecting_evidence"
        self.last_task = task_now

    def _validate_time(self, task_now):
        if (
            not math.isfinite(task_now)
            or task_now < self.start
            or (self.last_task is not None and task_now <= self.last_task)
        ):
            raise ValueError("fresh task time required")

    def restart(self, factory=None):
        if self.status != "pending":
            raise RuntimeError("release window is terminal")
        # Invalidate before cleanup: even a failed restart cannot retain evidence.
        self._invalidate("awaiting_clock_mapping")
        self.session.restart(factory)
        self.session_id = self.session.session

    def step(self, task_now, *, stop=False):
        if self.status != "pending":
            return self.status
        if stop:
            self._invalidate("user_stop")
            self.status = "cancelled"
            return self.status
        self._validate_time(task_now)
        self.last_task = task_now
        if task_now >= self.start + 3:
            self.status, self.reason = "unconfirmed", "delivery_deadline_reached"
            return self.status
        if self.session.session != self.session_id:
            self._invalidate("camera_session_changed")
            self.session_id = self.session.session
        observation = self.session.poll()
        if self.session.status in {"failed", "timed_out", "closed", "cleanup_failed"}:
            self._invalidate("camera_unavailable")
            return self.status
        if self.confirmation is None:
            return self.status
        if self.session.status in {"ready", "error"}:
            self.confirmation.step(task_now, self.clock(), observation)
            self.status, self.reason = self.confirmation.status, self.confirmation.reason
            self.rows.append(
                {
                    "event": "observation",
                    "session": self.session_id,
                    "task_time": task_now,
                    "capture_error": self.session.error,
                    **self.confirmation.result(),
                }
            )
        if self.status == "pending":
            self.session.request()
        return self.status

    def close(self):
        if self.status == "pending":
            self._invalidate("observer_closed")
            self.status = "cancelled"
        self.session.close()

    def result(self):
        return {
            "schema_version": 1,
            "status": self.status,
            "reason": self.reason,
            "session": self.session_id,
            "deadline": self.start + 3,
            "mapping_bound": self.confirmation is not None,
        }
