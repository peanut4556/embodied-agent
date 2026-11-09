"""Single-flight request controller for the delivery-aware visual challenge."""

import copy
import math
from dataclasses import asdict, dataclass

from .delivered_challenge import DeliveredChallenge
from .observation_capture import StampedPosition


@dataclass(frozen=True)
class PairCaptureRequest:
    request_id: int
    session_id: str
    task_time: float
    host_time: float


class SerialChallengeController:
    """Single controller thread; poll requests work, complete delivers one pair.

    No driver/thread is started here. The caller must own and cancel its driver.
    Terminal requests are quarantined, never replaced by another request while
    the old work may still run. No exposure predating its request is accepted.
    """

    def __init__(
        self,
        start_time,
        clock_map,
        session_id,
        *,
        position_error_m,
        delivery_budget_s=0.0,
        request_timeout_s=0.25,
    ):
        if (
            isinstance(request_timeout_s, bool)
            or not isinstance(request_timeout_s, (int, float))
            or not math.isfinite(request_timeout_s)
            or request_timeout_s <= 0
        ):
            raise ValueError("positive finite request timeout required")
        self.check = DeliveredChallenge(
            start_time,
            clock_map,
            session_id,
            position_error_m=position_error_m,
            delivery_budget_s=delivery_budget_s,
        )
        self.timeout = request_timeout_s
        self.status, self.reason = "pending", "awaiting_capture_request"
        self.inflight = None
        self.last_task = self.last_host = None
        self.requests, self.events = [], []
        self.ignored_results = 0

    def _fail(self, reason, *, status="unconfirmed"):
        self.status, self.reason = status, reason

    def _clock(self, now, host):
        if (
            not math.isfinite(now)
            or not math.isfinite(host)
            or now < self.check.start
            or (self.last_task is not None and now < self.last_task)
            or (self.last_host is not None and host < self.last_host)
        ):
            raise ValueError("monotonic finite controller clocks required")
        self.last_task, self.last_host = now, host
        mapping = self.check.clock_map
        predicted = mapping.task_origin + (host - mapping.host_origin) * mapping.rate
        if host < mapping.host_origin or abs(predicted - now) > mapping.max_error:
            self._fail("controller_clock_mismatch")
            return False
        return True

    def _sync(self):
        if self.check.status != "pending":
            self.status, self.reason = self.check.status, self.check.reason

    def poll(self, task_now, host_now, *, stop=False):
        if self.status != "pending":
            return None
        if stop:
            self._fail("user_stop", status="cancelled")
            return None
        if not self._clock(task_now, host_now):
            return None
        # A zero-latency final request may be issued exactly at both deadlines.
        # If it remains outstanding, the next poll expires it; completion at the
        # boundary must be delivered before that poll.
        boundary_request = (
            self.inflight is None
            and abs(task_now - self.check.deadline) <= 1e-9
            and task_now <= self.check.capture_deadline + 1e-9
            and self.check.next_capture_time <= task_now + 1e-9
        )
        if not boundary_request:
            self.check.tick(task_now)
        self._sync()
        if self.status != "pending":
            return None
        if self.inflight is not None:
            if host_now - self.inflight.host_time >= self.timeout - 1e-9:
                self._fail("capture_request_timeout")
            return None
        if task_now > self.check.capture_deadline + 1e-9:
            self._fail("capture_deadline_missed")
            return None
        if task_now < self.check.next_capture_time - 1e-9:
            return None
        if len(self.requests) >= 31:
            self._fail("capture_request_budget_exhausted")
            return None
        self.inflight = PairCaptureRequest(
            len(self.requests) + 1, self.check.session, task_now, host_now
        )
        self.requests.append(asdict(self.inflight))
        return self.inflight

    def complete(self, request_id, task_now, host_now, primary, secondary, *, session_id):
        """Return True when the current request was consumed, False when quarantined."""
        if (
            self.status != "pending"
            or self.inflight is None
            or request_id != self.inflight.request_id
        ):
            self.ignored_results += 1
            return False
        if not self._clock(task_now, host_now):
            return False
        request = self.inflight
        if host_now - request.host_time >= self.timeout - 1e-9:
            self._fail("capture_request_timeout")
            return False
        self.inflight = None
        if any(not isinstance(p, StampedPosition) for p in (primary, secondary)):
            self._fail("missing_timestamped_pair")
        elif any(
            isinstance(p.captured_at, bool)
            or not isinstance(p.captured_at, (int, float))
            or not math.isfinite(p.captured_at)
            or p.captured_at < request.host_time - 1e-9
            or p.captured_at > host_now
            for p in (primary, secondary)
        ):
            self._fail("capture_outside_request_interval")
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
                "inflight": None if self.inflight is None else asdict(self.inflight),
                "driver_cancel_required": self.status != "pending" and self.inflight is not None,
                "request_count": len(self.requests),
                "ignored_results": self.ignored_results,
                "requests": self.requests,
                "events": self.events,
                "confirmation": self.check.result(),
            }
        )
