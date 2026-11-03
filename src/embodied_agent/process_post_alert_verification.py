"""Process acquisition for monitoring and one bounded post-alert recheck."""

import copy
import math
import time

from .post_alert_verification import PostAlertVerification
from .process_post_confirmation import ProcessPostConfirmation


class ProcessPostAlertVerification(ProcessPostConfirmation):
    """Reuse explicit handoff, ownership, clock guards and cleanup semantics.

    Unlike the base adapter, the live phases include rechecking. No restart or
    motion commands; only poll/request in step, and process reaping in close.
    """

    def __init__(
        self,
        session,
        confirmation_result,
        start_time,
        clock_map,
        *,
        duration=2.0,
        clock=time.monotonic,
    ):
        supervisor = PostAlertVerification(
            confirmation_result, start_time, session.session, clock_map, duration=duration
        )
        super().__init__(
            session, confirmation_result, start_time, clock_map, duration=duration, clock=clock
        )
        self.watch = supervisor

    def step(self, task_now, *, stop=False):
        if self.status not in {"monitoring", "rechecking"}:
            return self.status
        if stop:
            return self.watch.step(task_now, None, None, session_id=self.session.session, stop=True)
        if (
            type(task_now) not in (int, float)
            or not math.isfinite(task_now)
            or task_now < self.watch.watch.start
            or (self.last_task is not None and task_now <= self.last_task)
        ):
            self.watch.invalidate("invalid_task_clock")
            return self.status
        self.last_task = task_now
        if (
            self.session.session != self._claim[0]
            or getattr(self.session, "_followup_claim", None) is not self._claim
        ):
            self.watch.invalidate("camera_ownership_changed")
            return self.status
        host = self._checked_host_time(task_now)
        if host is None:
            return self.status
        deadline = self.watch.recheck.start + 3 if self.watch.recheck else self.watch.watch.deadline
        if task_now >= deadline:
            # Do not acquire or consume a buffered frame after the adapter deadline.
            return self.watch.step(task_now, host, None, session_id=self.session.session)
        pending = self.session.status == "pending"
        observation = self.session.poll()
        host = self._checked_host_time(task_now)
        if host is None:
            return self.status
        if (
            self.session.session != self._claim[0]
            or getattr(self.session, "_followup_claim", None) is not self._claim
        ):
            self.watch.invalidate("camera_ownership_changed")
            return self.status
        if self.session.status in {"failed", "timed_out", "closed", "cleanup_failed"}:
            self.watch.invalidate("camera_unavailable")
            self.rows.append(
                {
                    "task_time": task_now,
                    "camera_status": self.session.status,
                    "error": self.session.error,
                }
            )
            return self.status
        delivered = pending and self.session.status in {"ready", "error"}
        expired = (
            self.last_valid_capture is not None
            and host - self.last_valid_capture > self.watch.clock_map.max_age
        )
        previous = (
            self.watch.recheck.last_exposure_time
            if self.watch.recheck
            else self.watch.watch.last_exposure
        )
        self.watch.step(
            task_now,
            host,
            observation if delivered else None,
            session_id=self.session.session,
            waiting=not delivered and not expired,
        )
        if delivered or expired:
            # Track only accepted frames. A rejected result must break continuity.
            accepted = (
                self.watch.recheck.last_exposure_time
                if self.watch.recheck
                else self.watch.watch.last_exposure
            )
            self.last_valid_capture = (
                observation.captured_at
                if delivered
                and observation is not None
                and accepted is not None
                and accepted != previous
                and self.watch.last_error is None
                and (self.watch.recheck is None or self.watch.recheck.last_error is None)
                else None
            )
            self.rows.append(
                {
                    "task_time": task_now,
                    "event": "delivery" if delivered else "expired",
                    "result": self.watch.result(),
                }
            )
        if (
            self.status in {"monitoring", "rechecking"}
            and self.session.status in {"idle", "ready", "error"}
            and not self.session.request()
        ):
            self.watch.invalidate("camera_request_failed")
        return self.status

    def close(self):
        if self.status in {"monitoring", "rechecking"}:
            self.watch.step(None, None, None, session_id=self.session.session, stop=True)
        super().close()

    def result(self):
        return {
            **self.watch.result(),
            "cleanup": copy.deepcopy(self.cleanup),
            "cleanup_required": self.cleanup["status"] != "succeeded",
        }
