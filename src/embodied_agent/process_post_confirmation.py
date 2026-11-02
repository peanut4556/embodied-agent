"""Explicit handoff from release confirmation to a bounded process camera watch."""

import copy
import math
import time

from .post_confirmation_watch import PostConfirmationWatch


class ProcessPostConfirmation:
    """Single owner. step never joins; close/context exit reaps the camera process.

    After handoff, the original observer must no longer operate/close the session.
    This adapter does not restart devices or send mechanical commands.
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
        if session.status not in {"idle", "ready"}:
            raise ValueError("idle camera with no pending request required")
        claim = getattr(session, "_followup_claim", None)
        if claim is not None and claim[0] == session.session:
            raise RuntimeError("camera session already assigned to a follow-up")
        self.watch = PostConfirmationWatch(
            confirmation_result, start_time, session.session, clock_map, duration=duration
        )
        self.session, self.clock = session, clock
        self.last_task = None
        self.last_valid_capture = None
        self.cleanup = {"status": "not_requested", "error": None}
        self.rows = []
        self._claim = (session.session, object())
        session._followup_claim = self._claim

    @classmethod
    def from_confirmed(cls, observer, *, duration=2.0):
        if observer.session_transferred:
            raise RuntimeError("observer camera session already transferred")
        if (
            observer.status != "confirmed_in_tray"
            or observer.confirmation is None
            or observer.last_task is None
            or observer.session_id != observer.session.session
        ):
            raise ValueError("current successful observer session required")
        followup = cls(
            observer.session,
            observer.result(),
            observer.last_task,
            observer.confirmation.clock_map,
            duration=duration,
            clock=observer.clock,
        )
        observer.session_transferred = True
        return followup

    @property
    def status(self):
        return self.watch.status

    def step(self, task_now, *, stop=False):
        if self.status != "monitoring":
            return self.status
        if stop:
            return self.watch.step(task_now, None, None, session_id=self.session.session, stop=True)
        if (
            not math.isfinite(task_now)
            or task_now < self.watch.start
            or (self.last_task is not None and task_now <= self.last_task)
        ):
            raise ValueError("fresh process follow-up task time required")
        self.last_task = task_now
        # Stop/expiry/session changes must be handled before any IPC or new request.
        if getattr(self.session, "_followup_claim", None) is not self._claim:
            self.watch.invalidate("camera_ownership_changed")
            return self.status
        if self.session.session != self.watch.session or task_now >= self.watch.deadline:
            return self.watch.step(task_now, None, None, session_id=self.session.session)
        pending = self.session.status == "pending"
        observation = self.session.poll()
        host_now = self.clock()
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
        if pending and self.session.status in {"ready", "error"}:
            previous_exposure = self.watch.last_exposure
            self.watch.step(task_now, host_now, observation, session_id=self.session.session)
            if self.watch.last_exposure != previous_exposure:
                self.last_valid_capture = observation.captured_at
            else:
                self.last_valid_capture = None
            self.rows.append(
                {
                    "task_time": task_now,
                    "camera_status": self.session.status,
                    "error": self.session.error,
                    "watch": self.watch.result(),
                }
            )
        elif (
            self.last_valid_capture is not None
            and host_now - self.last_valid_capture > self.watch.clock_map.max_age
        ):
            # Polling a pending read is not a missing frame, but stale evidence
            # must not continue to display an inside/outside classification.
            self.watch.step(task_now, host_now, None, session_id=self.session.session)
            self.last_valid_capture = None
        if (
            self.status == "monitoring"
            and self.session.status in {"idle", "ready", "error"}
            and not self.session.request()
        ):
            self.watch.invalidate("camera_request_failed")
        return self.status

    def close(self):
        if self.cleanup["status"] == "succeeded":
            return
        if self.status == "monitoring":
            self.watch.step(None, None, None, session_id=self.session.session, stop=True)
        try:
            if (
                self.session.session != self._claim[0]
                or getattr(self.session, "_followup_claim", None) is not self._claim
            ):
                raise RuntimeError("camera ownership changed; refusing to close another session")
            self.session.close()
        except Exception as exc:
            self.cleanup = {"status": "failed", "error": str(exc)[:512]}
            raise
        else:
            self.cleanup = {"status": "succeeded", "error": None}

    def result(self):
        return {
            **self.watch.result(),
            "cleanup": copy.deepcopy(self.cleanup),
            "cleanup_required": self.cleanup["status"] != "succeeded",
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if exc is not None:
                raise BaseExceptionGroup(
                    "follow-up body and camera cleanup failed", [exc, cleanup_error]
                ) from None
            raise
