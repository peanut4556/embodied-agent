"""Experimental dual-clock confirmation. No automatic clock calibration."""

import math
from dataclasses import dataclass

from .observation_capture import StampedPosition
from .release_confirmation import ReleaseConfirmation


@dataclass(frozen=True)
class ExposureClockMap:
    host_origin: float
    task_origin: float
    rate: float = 1.0
    max_error: float = 0.02
    max_age: float = 0.25

    def __post_init__(self):
        if (
            not all(
                math.isfinite(v)
                for v in (
                    self.host_origin,
                    self.task_origin,
                    self.rate,
                    self.max_error,
                    self.max_age,
                )
            )
            or self.rate <= 0
            or self.max_error < 0
            or self.max_age <= 0
        ):
            raise ValueError("invalid fixed clock mapping")

    def map(self, captured_at, host_now, task_now):
        if not all(math.isfinite(v) for v in (captured_at, host_now, task_now)):
            raise ValueError("non-finite observation clock")
        if not self.host_origin <= captured_at <= host_now:
            raise ValueError("capture outside mapped clock interval")
        if host_now - captured_at > self.max_age:
            raise ValueError("capture too old")
        predicted_now = self.task_origin + (host_now - self.host_origin) * self.rate
        if abs(predicted_now - task_now) > self.max_error:
            raise ValueError("clock mapping drift exceeds tolerance")
        mapped = self.task_origin + (captured_at - self.host_origin) * self.rate
        if mapped > task_now:
            raise ValueError("mapped capture is in the future")
        return mapped


class ExposureConfirmation:
    """Evidence uses mapped exposure time; the deadline uses delivery task time."""

    def __init__(self, start_time, clock_map):
        self.evidence = ReleaseConfirmation(start_time)
        self.start = start_time
        self.clock_map = clock_map
        self.last_delivery = None
        self.status = "pending"
        self.reason = "collecting_evidence"
        self.last_error = None
        self.last_exposure_time = None
        self.last_accepted_exposure = None

    def step(self, task_now, host_now, observation, *, stop=False):
        if self.status != "pending":
            return self.status
        if stop:
            self.status, self.reason = "cancelled", "user_stop"
            return self.status
        if (
            not math.isfinite(task_now)
            or task_now < self.start
            or (self.last_delivery is not None and task_now <= self.last_delivery)
        ):
            raise ValueError("fresh task delivery time required")
        self.last_delivery = task_now
        if task_now - self.start > 3 + 1e-9:
            self.status, self.reason = "unconfirmed", "delivery_deadline_exceeded"
            return self.status
        self.last_error = None
        self.last_exposure_time = None
        valid = False
        if observation is not None:
            try:
                if not isinstance(observation, StampedPosition):
                    raise TypeError("timestamped position required")
                exposure = self.clock_map.map(observation.captured_at, host_now, task_now)
                if (
                    self.last_accepted_exposure is not None
                    and exposure <= self.last_accepted_exposure
                ):
                    raise ValueError("fresh exposure time required")
                self.evidence.observe(exposure, observation.position)
                self.last_exposure_time = exposure
                self.last_accepted_exposure = exposure
                valid = True
            except (ValueError, TypeError) as exc:
                self.last_error = str(exc)
        if not valid:
            # Missing/rejected evidence breaks continuity, without moving the deadline.
            self.evidence = ReleaseConfirmation(self.start)
        if self.evidence.status == "confirmed_in_tray":
            self.status, self.reason = self.evidence.status, self.evidence.reason
        elif task_now - self.start >= 3 - 1e-9:
            if self.last_error:
                self.status, self.reason = "unconfirmed", "observation_rejected"
            elif observation is None:
                self.status, self.reason = "unconfirmed", "missing_observation"
            else:
                # Apply existing spatial/timeout semantics at the delivery deadline.
                final = ReleaseConfirmation(self.start)
                final.observe(task_now, observation.position)
                self.status, self.reason = final.status, final.reason
        return self.status

    def result(self):
        return {
            "schema_version": 1,
            "status": self.status,
            "reason": self.reason,
            "last_error": self.last_error,
            "last_exposure_time": self.last_exposure_time,
        }
