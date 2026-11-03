"""One bounded visual recheck after a sustained post-confirmation alert."""

import copy
import math

from .exposure_confirmation import ExposureConfirmation
from .observation_capture import StampedPosition
from .post_confirmation_watch import PostConfirmationWatch


class PostAlertVerification:
    """Consume delivered frames, never command motion or acquire/restart devices.

    A recheck starts at alert delivery, excludes all earlier exposures, and has
    one fixed three-second budget. A reconfirmation is a historical observation,
    not continuous safety or proof that a physical recovery action succeeded.
    """

    def __init__(self, confirmation_result, start_time, session_id, clock_map, *, duration=2):
        self.watch = PostConfirmationWatch(
            confirmation_result, start_time, session_id, clock_map, duration=duration
        )
        if not math.isfinite(self.watch.deadline + 3):
            raise ValueError("finite verification deadline required")
        self.status, self.reason = "monitoring", "awaiting_followup_observation"
        self.recheck = None
        self.last_task = self.last_host = None
        self.last_error = None
        self.events = []

    def _transition(self, status, reason, task_now):
        self.status, self.reason = status, reason
        self.events.append({"status": status, "reason": reason, "task_time": task_now})

    def step(self, task_now, host_now, observation, *, session_id, stop=False):
        if self.status not in {"monitoring", "rechecking"}:
            return self.status
        if stop:
            # Stop does not depend on a valid clock or frame.
            self._transition("cancelled", "user_stop", self.last_task)
            return self.status
        if (
            type(task_now) not in (int, float)
            or not math.isfinite(task_now)
            or task_now < self.watch.start
            or (self.last_task is not None and task_now <= self.last_task)
        ):
            self._transition("invalidated", "invalid_task_clock", self.last_task)
            return self.status
        self.last_task = task_now
        if session_id != self.watch.session:
            self._transition("invalidated", "camera_session_changed", task_now)
            return self.status
        mapping = self.watch.clock_map
        if (
            type(host_now) not in (int, float)
            or not math.isfinite(host_now)
            or host_now < mapping.host_origin
            or (self.last_host is not None and host_now < self.last_host)
        ):
            self._transition("invalidated", "invalid_host_clock", task_now)
            return self.status
        predicted = mapping.task_origin + (host_now - mapping.host_origin) * mapping.rate
        if not math.isfinite(predicted) or abs(predicted - task_now) > mapping.max_error:
            self._transition("invalidated", "clock_mapping_drift", task_now)
            return self.status
        self.last_host = host_now
        self.last_error = None
        if self.status == "monitoring":
            self.watch.step(task_now, host_now, observation, session_id=session_id)
            if self.watch.alerts:
                self.recheck = ExposureConfirmation(task_now, mapping)
                self._transition("rechecking", "sustained_outside_alert", task_now)
            elif self.watch.status == "completed":
                self._transition("completed_without_alert", "followup_window_ended", task_now)
            return self.status

        # The alarm frame and delayed pre-alert frames cannot seed new evidence.
        if observation is not None:
            try:
                if not isinstance(observation, StampedPosition):
                    raise TypeError("timestamped position required")
                exposure = mapping.map(observation.captured_at, host_now, task_now)
                if exposure <= self.recheck.start:
                    raise ValueError("post-alert exposure required")
                point = tuple(float(value) for value in observation.position)
                if len(point) != 3 or not all(math.isfinite(value) for value in point):
                    raise ValueError("finite 3D position required")
                observation = StampedPosition(point, observation.captured_at)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                self.last_error = str(exc)[:512]
                observation = None
        self.recheck.step(task_now, host_now, observation)
        if self.recheck.status == "confirmed_in_tray":
            self._transition("reconfirmed", "fresh_stable_visual_evidence", task_now)
        elif self.recheck.status != "pending":
            self._transition("needs_review", self.recheck.reason, task_now)
        return self.status

    def result(self):
        return copy.deepcopy(
            {
                "schema_version": 1,
                "status": self.status,
                "reason": self.reason,
                "last_error": self.last_error,
                "original_confirmation": self.watch.original_confirmation,
                "alert": self.watch.alerts[0] if self.watch.alerts else None,
                "recheck_start": self.recheck.start if self.recheck else None,
                "deadline": self.recheck.start + 3 if self.recheck else self.watch.deadline,
                "recheck": self.recheck.result() if self.recheck else None,
                "events": self.events,
            }
        )
