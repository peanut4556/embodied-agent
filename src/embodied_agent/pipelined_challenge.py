"""Bounded speculative capture schedule; driver concurrency remains external."""

import copy
import math
from dataclasses import asdict

from .observation_capture import StampedPosition
from .serial_challenge import PairCaptureRequest, SerialChallengeController


class PipelinedChallengeController(SerialChallengeController):
    """Reuse clock/deadline validation, but replace the single-flight protocol.

    Request 5 Hz through two seconds, then speculatively request 20 Hz without
    waiting for candidate delivery. No result buffering or driver threads here.
    Complete in request order before polling at the same instant. Missing a slot
    fails closed; no catch-up burst, dropped frame, or shifted exposure window.
    Optional lateness only widens admission within the original capture deadline;
    evidence still uses actual exposure times and its unchanged cadence rules.
    """

    def __init__(self, *args, capacity=2, exposure_lateness_s=0.0, **kwargs):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 31:
            raise ValueError("capacity must be an integer between 1 and 31")
        if (
            isinstance(exposure_lateness_s, bool)
            or not isinstance(exposure_lateness_s, (int, float))
            or not math.isfinite(exposure_lateness_s)
            or not 0 <= exposure_lateness_s <= 0.01
        ):
            raise ValueError("exposure lateness must be finite and between 0 and 0.01 seconds")
        super().__init__(*args, **kwargs)
        self.exposure_lateness_s = float(exposure_lateness_s)
        self.capacity = capacity
        self.pending = {}
        self.peak_inflight = 0
        self.schedule = tuple(
            self.check.start + ms / 1000 for ms in (*range(0, 2001, 200), *range(2050, 3001, 50))
        )

    def _expire(self, host_now):
        if any(host_now - r.host_time >= self.timeout - 1e-9 for r in self.pending.values()):
            self._fail("capture_request_timeout")
            return True
        return False

    def poll(self, task_now, host_now, *, stop=False):
        if self.status != "pending":
            return None
        if stop:
            self._fail("user_stop", status="cancelled")
            return None
        if not self._clock(task_now, host_now) or self._expire(host_now):
            return None
        index = len(self.requests)
        due = self.schedule[index] if index < len(self.schedule) else None
        # Allow an immediate final completion at a zero-reserve deadline.
        if due is None or abs(task_now - self.check.deadline) > 1e-9 or abs(task_now - due) > 1e-9:
            self.check.tick(task_now)
            self._sync()
        if self.status != "pending" or due is None or task_now < due - 1e-9:
            return None
        if task_now > min(due + self.exposure_lateness_s, self.check.capture_deadline) + 1e-9:
            self._fail("capture_schedule_missed")
            return None
        if len(self.pending) >= self.capacity:
            self._fail("pipeline_capacity_exhausted")
            return None
        request = PairCaptureRequest(index + 1, self.check.session, task_now, host_now)
        self.pending[request.request_id] = request
        self.requests.append(asdict(request))
        self.peak_inflight = max(self.peak_inflight, len(self.pending))
        return request

    def complete(self, request_id, task_now, host_now, primary, secondary, *, session_id):
        if self.status != "pending" or request_id not in self.pending:
            self.ignored_results += 1
            return False
        if not self._clock(task_now, host_now) or self._expire(host_now):
            return False
        if request_id != next(iter(self.pending)):
            self._fail("out_of_order_completion")
            return False
        request = self.pending.pop(request_id)
        due = self.schedule[request_id - 1]
        latest = min(due + self.exposure_lateness_s, self.check.capture_deadline)
        mapping = self.check.clock_map
        if any(not isinstance(p, StampedPosition) for p in (primary, secondary)):
            self._fail("missing_timestamped_pair")
        elif any(
            isinstance(p.captured_at, bool)
            or not isinstance(p.captured_at, (int, float))
            or not math.isfinite(p.captured_at)
            or (self.exposure_lateness_s == 0 and abs(p.captured_at - request.host_time) > 1e-9)
            or p.captured_at < request.host_time - 1e-9
            or mapping.task_origin + (p.captured_at - mapping.host_origin) * mapping.rate
            > latest + 1e-9
            or p.captured_at > host_now
            for p in (primary, secondary)
        ):
            self._fail("capture_not_at_scheduled_request")
        else:
            self.check.deliver(task_now, host_now, primary, secondary, session_id=session_id)
            self._sync()
            if self.status == "pending":
                self.reason = self.check.reason
        self.events.append(
            {
                "request_id": request_id,
                "delivery_time": task_now,
                "status": self.status,
                "reason": self.reason,
            }
        )
        return True

    def result(self):
        return copy.deepcopy(
            {
                "status": self.status,
                "reason": self.reason,
                "capacity": self.capacity,
                "exposure_lateness_s": self.exposure_lateness_s,
                "peak_inflight": self.peak_inflight,
                "pending": [asdict(r) for r in self.pending.values()],
                "driver_cancel_required": self.status != "pending" and bool(self.pending),
                "cancel_request_ids": list(self.pending) if self.status != "pending" else [],
                "request_count": len(self.requests),
                "ignored_results": self.ignored_results,
                "requests": self.requests,
                "events": self.events,
                "confirmation": self.check.result(),
            }
        )
