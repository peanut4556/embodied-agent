"""Bounded visual follow-up; outside the confirmation region is not a fall verdict."""

import copy
import math

from .exposure_confirmation import ExposureClockMap
from .observation_capture import StampedPosition


class PostConfirmationWatch:
    """Explicit follow-up consuming delivered frames, with no device/control actions.

    Call step even when capture is missing; a result is a snapshot at that call,
    not a guarantee that the object remains inside between observations.
    """

    def __init__(self, confirmation_result, start_time, session_id, clock_map, *, duration=2.0):
        if confirmation_result.get("status") != "confirmed_in_tray":
            raise ValueError("successful visual confirmation required")
        if (
            not math.isfinite(start_time)
            or not math.isfinite(duration)
            or duration <= 0
            or not math.isfinite(start_time + duration)
        ):
            raise ValueError("finite start and positive duration required")
        if (
            not isinstance(clock_map, ExposureClockMap)
            or not isinstance(session_id, str)
            or not session_id
        ):
            raise ValueError("explicit clock mapping and session required")
        self.original_confirmation = copy.deepcopy(confirmation_result)
        self.start, self.deadline = start_time, start_time + duration
        self.session, self.clock_map = session_id, clock_map
        self.status, self.reason = "monitoring", "awaiting_followup_observation"
        self.evidence_state = "unknown"
        self.last_task = self.last_exposure = None
        self.outside_since = self.outside_previous = None
        self.last_error = None
        self.alerts = []
        self.rows = []

    def _unknown(self, reason):
        self.evidence_state, self.reason = "unknown", reason
        self.outside_since = self.outside_previous = None

    def step(self, task_now, host_now, observation, *, session_id, stop=False):
        if self.status != "monitoring":
            return self.status
        if stop:
            self.status = "cancelled"
            self._unknown("user_stop")
            return self.status
        if (
            not math.isfinite(task_now)
            or task_now < self.start
            or (self.last_task is not None and task_now <= self.last_task)
        ):
            raise ValueError("fresh follow-up task time required")
        self.last_task = task_now
        self.last_error = None
        if session_id != self.session:
            self.status = "invalidated"
            self._unknown("camera_session_changed")
            return self.status
        if task_now >= self.deadline:
            self.status = "completed"
            self._unknown("followup_window_ended")
            return self.status
        if observation is None:
            self._unknown("missing_observation")
        else:
            try:
                if not isinstance(observation, StampedPosition):
                    raise TypeError("timestamped follow-up position required")
                exposure = self.clock_map.map(observation.captured_at, host_now, task_now)
                if exposure <= self.start or (
                    self.last_exposure is not None and exposure <= self.last_exposure
                ):
                    raise ValueError("fresh post-confirmation exposure required")
                point = tuple(float(x) for x in observation.position)
                if len(point) != 3 or not all(math.isfinite(x) for x in point):
                    raise ValueError("finite 3D position required")
                self.last_exposure = exposure
                inside = (
                    abs(point[0] - 0.64) < 0.05
                    and abs(point[1]) < 0.04
                    and 0.032 < point[2] < 0.058
                )
                if inside:
                    self.outside_since = self.outside_previous = None
                    self.evidence_state, self.reason = "inside_region", "fresh_inside_observation"
                else:
                    if self.outside_previous is None or exposure - self.outside_previous > 0.25:
                        self.outside_since = exposure
                    self.outside_previous = exposure
                    sustained = exposure - self.outside_since >= 0.25 - 1e-9
                    self.evidence_state = "outside_region" if sustained else "outside_candidate"
                    self.reason = (
                        "sustained_outside_observations"
                        if sustained
                        else "single_outside_observation"
                    )
                    if sustained and not self.alerts:
                        self.alerts.append(
                            {
                                "event": "outside_confirmation_region_detected",
                                "task_time": task_now,
                                "exposure_time": exposure,
                            }
                        )
            except (ValueError, TypeError, OverflowError) as exc:
                self.last_error = str(exc)
                self._unknown("observation_rejected")
        self.rows.append(
            {
                "task_time": task_now,
                "evidence_state": self.evidence_state,
                "reason": self.reason,
                "last_error": self.last_error,
            }
        )
        return self.status

    def result(self):
        return copy.deepcopy(
            {
                "schema_version": 1,
                "status": self.status,
                "reason": self.reason,
                "evidence_state": self.evidence_state,
                "last_exposure_time": self.last_exposure,
                "last_error": self.last_error,
                "deadline": self.deadline,
                "session": self.session,
                "outside_detected": bool(self.alerts),
                "alerts": self.alerts,
                "original_confirmation": self.original_confirmation,
            }
        )
