"""Bounded, explicitly dispatched recovery for an experimental release observer."""

import math

from .process_exposure_observer import ProcessExposureObserver


class RecoveryExposureObserver(ProcessExposureObserver):
    """Call restart only from lifecycle management, with a live task clock.

    Recovery never calibrates clocks or issues mechanical commands. Startup,
    mapping and evidence must still fit in the original release window.
    """

    def __init__(
        self, session, start_time, *, task_clock, max_restarts=1, min_remaining=1.25, **kwargs
    ):
        if type(max_restarts) is not int or max_restarts < 0:
            raise ValueError("nonnegative integer restart limit required")
        if not math.isfinite(min_remaining) or min_remaining < 1:
            raise ValueError("recovery must reserve at least one second for evidence")
        super().__init__(session, start_time, **kwargs)
        self.task_clock = task_clock
        self.max_restarts = max_restarts
        self.min_remaining = min_remaining
        self.attempts = []
        self.cleanup = {"status": "not_requested", "error": None}

    def _end_recovery(self, reason):
        self._invalidate(reason)
        self.status = "unconfirmed"
        if self.attempts and self.attempts[-1]["outcome"] == "awaiting_clock_mapping":
            self.attempts[-1]["outcome"] = reason

    def restart(self, factory=None):
        """Return whether spawn succeeded; this does not mean observation recovered."""
        if self.status != "pending":
            return False
        task_now = self.task_clock()
        self._validate_time(task_now)
        self.last_task = task_now
        if task_now >= self.start + 3:
            self._end_recovery("delivery_deadline_reached")
            return False
        if self.session.status == "cleanup_failed":
            self._end_recovery("camera_cleanup_failed")
            return False
        if self.session.status not in {"failed", "timed_out"}:
            return False
        if len(self.attempts) >= self.max_restarts:
            self._end_recovery("restart_limit_reached")
            return False
        if self.start + 3 - task_now <= self.min_remaining:
            self._end_recovery("insufficient_recovery_budget")
            return False
        attempt = {
            "attempt": len(self.attempts) + 1,
            "task_time": task_now,
            "old_session": self.session.session,
            "fault": self.session.status,
            "error": self.session.error,
            "outcome": "restarting",
        }
        self.attempts.append(attempt)
        try:
            super().restart(factory)
        except Exception as exc:  # noqa: BLE001 - lifecycle boundary, preserve failure evidence
            attempt.update(outcome="restart_failed", restart_error=str(exc)[:512])
            self._end_recovery("camera_restart_failed")
            return False
        attempt.update(new_session=self.session.session, outcome="awaiting_clock_mapping")
        # Cleanup/spawn consumes real task time; never reuse the pre-restart reading.
        completed_at = self.task_clock()
        if not math.isfinite(completed_at) or completed_at < task_now:
            attempt["outcome"] = "invalid_task_clock"
            self._end_recovery("invalid_recovery_clock")
            return False
        self.last_task = completed_at
        attempt["completed_at"] = completed_at
        if completed_at >= self.start + 3:
            attempt["outcome"] = "deadline_exceeded"
            self._end_recovery("delivery_deadline_reached")
            return False
        return True

    def step(self, task_now, *, stop=False):
        status = super().step(task_now, stop=stop)
        if self.attempts:
            attempt = self.attempts[-1]
            if attempt["outcome"] == "awaiting_clock_mapping":
                if (
                    self.confirmation is not None
                    and self.confirmation.last_exposure_time is not None
                ):
                    attempt["outcome"] = "observation_restored"
                elif status != "pending":
                    attempt["outcome"] = status
                elif self.reason == "camera_unavailable":
                    attempt["outcome"] = "camera_failed_again"
        return status

    def result(self):
        return {
            **super().result(),
            "schema_version": 2,
            "max_restarts": self.max_restarts,
            "restart_count": len(self.attempts),
            "recovery_attempts": [dict(attempt) for attempt in self.attempts],
        }

    def close(self):
        try:
            super().close()
        except Exception as exc:
            self.cleanup = {"status": "failed", "error": str(exc)[:512]}
            if self.attempts and self.attempts[-1]["outcome"] == "awaiting_clock_mapping":
                self.attempts[-1]["outcome"] = "cleanup_failed"
            raise
        else:
            self.cleanup = {"status": "succeeded", "error": None}
        finally:
            if self.attempts and self.attempts[-1]["outcome"] == "awaiting_clock_mapping":
                self.attempts[-1]["outcome"] = "observer_closed"
