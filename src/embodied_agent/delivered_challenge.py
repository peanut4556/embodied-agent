"""Delivery-clock boundary for paired exposure-time confirmation challenges."""

import copy
import math

from .confirmation_challenge import ConfirmationChallenge
from .exposure_confirmation import ExposureClockMap
from .observation_capture import StampedPosition


class DeliveredChallenge:
    """Fail closed on invalid evidence; enforce the deadline even without frames.

    Call deliver for one synchronized pair, and tick while no pair is available.
    The default deadline is three seconds; an explicit delivery budget may add
    up to 250 ms for returning evidence captured within the original window.
    Both are single-controller operations. Capture scheduling/driver ownership
    remain external; this boundary never fabricates a capture or extends the configured deadline.
    """

    def __init__(
        self, start_time, clock_map, session_id, *, position_error_m, delivery_budget_s=0.0
    ):
        if not isinstance(clock_map, ExposureClockMap):
            raise TypeError("explicit exposure clock map required")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("nonempty camera session required")
        if (
            isinstance(delivery_budget_s, bool)
            or not isinstance(delivery_budget_s, (int, float))
            or not math.isfinite(delivery_budget_s)
            or not 0 <= delivery_budget_s <= 0.25
        ):
            raise ValueError("delivery budget must be finite and between 0 and 0.25 seconds")
        self.delivery_budget_s = float(delivery_budget_s)
        self.capture_deadline = float(start_time) + 3
        self.evidence = ConfirmationChallenge(start_time, position_error_m=position_error_m)
        self.clock_map, self.session = clock_map, session_id
        self.start, self.deadline = (
            float(start_time),
            self.capture_deadline + self.delivery_budget_s,
        )
        self.status, self.reason = "pending", "awaiting_delivery"
        self.last_task = self.last_delivery = None
        self.rows = []

    @property
    def next_capture_time(self):
        return self.evidence.next_capture_time if self.status == "pending" else None

    def _time(self, now):
        if (
            not math.isfinite(now)
            or now < self.start
            or (self.last_task is not None and now < self.last_task)
        ):
            raise ValueError("monotonic task time required")

    def _end(self, reason, *, status="unconfirmed"):
        self.status, self.reason = status, reason
        return self.status

    def tick(self, task_now, *, stop=False):
        if self.status != "pending":
            return self.status
        if stop:
            return self._end("user_stop", status="cancelled")
        self._time(task_now)
        self.last_task = task_now
        if task_now >= self.deadline - 1e-9:
            return self._end("delivery_deadline_without_complete_evidence")
        return self.status

    def deliver(self, task_now, host_now, primary, secondary, *, session_id, stop=False):
        if self.status != "pending":
            return self.status
        if stop:
            return self._end("user_stop", status="cancelled")
        self._time(task_now)
        if self.last_delivery is not None and task_now <= self.last_delivery:
            raise ValueError("fresh delivery time required")
        self.last_task = self.last_delivery = task_now
        exposure = None
        error = None
        if task_now > self.deadline + 1e-9:
            self._end("delivery_deadline_exceeded")
        elif session_id != self.session:
            self._end("camera_session_changed")
        else:
            try:
                if not isinstance(primary, StampedPosition) or not isinstance(
                    secondary, StampedPosition
                ):
                    raise TypeError("two timestamped positions required")
                times = [
                    self.clock_map.map(p.captured_at, host_now, task_now)
                    for p in (primary, secondary)
                ]
                if abs(times[0] - times[1]) > 1e-9:
                    raise ValueError("paired exposures must share a capture epoch")
                exposure = min(times)
                self.evidence.step(exposure, primary.position, secondary.position)
                self.status, self.reason = self.evidence.status, self.evidence.reason
            except (TypeError, ValueError, OverflowError) as exc:
                error = str(exc)
                self._end("exposure_rejected")
        if self.status == "pending" and task_now >= self.deadline - 1e-9:
            self._end("delivery_deadline_without_complete_evidence")
        self.rows.append(
            {
                "delivery_time": task_now,
                "exposure_time": exposure,
                "status": self.status,
                "reason": self.reason,
                "error": error,
            }
        )
        return self.status

    def result(self):
        return copy.deepcopy(
            {
                "status": self.status,
                "reason": self.reason,
                "deadline": self.deadline,
                "capture_deadline": self.capture_deadline,
                "delivery_budget_s": self.delivery_budget_s,
                "session_id": self.session,
                "next_capture_time": self.next_capture_time,
                "deliveries": self.rows,
                "evidence": self.evidence.result(),
            }
        )
