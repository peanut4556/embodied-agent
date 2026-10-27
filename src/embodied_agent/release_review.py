"""Experimental, one-shot visual recheck; never commands robot motion."""

import numpy as np

from .release_confirmation import ReleaseConfirmation


class ReleaseReview:
    def __init__(self, start_time):
        self.initial = ReleaseConfirmation(start_time)
        self.recheck = None
        self.deadline = float(start_time) + 6
        self.next_time = float(start_time)
        self.last_time = None
        self.status = "pending"
        self.reason = "collecting_evidence"
        self.observation_count = 0

    def step(self, timestamp, locate, *, stop=False):
        if self.status != "pending":
            return self.status
        if stop:
            self.status, self.reason = "cancelled", "user_stop"
            return self.status
        if (
            not np.isfinite(timestamp)
            or timestamp < self.initial.start
            or (self.last_time is not None and timestamp <= self.last_time)
        ):
            raise ValueError("fresh finite observation time required")
        self.last_time = timestamp
        # A delayed caller cannot refresh the shared six-second budget.
        if timestamp > self.deadline + 1e-9:
            self.status, self.reason = "unconfirmed", "overall_timeout"
            return self.status
        if timestamp + 1e-9 < self.next_time and timestamp < self.deadline - 1e-9:
            return self.status
        try:
            point = locate()
        except ValueError:
            point = None
        self.observation_count += 1
        check = self.recheck if self.recheck is not None else self.initial
        check.observe(timestamp, point)
        if (
            check is self.initial
            and check.status == "unconfirmed"
            and check.reason in {"missing_observation", "insufficient_stable_evidence"}
            and timestamp < self.deadline
        ):
            # The first monitor tolerates sub-nanosecond boundary roundoff.
            self.recheck = ReleaseConfirmation(min(timestamp, self.initial.start + 3))
            self.recheck.observe(timestamp, point)
            check = self.recheck
        if check.status != "pending":
            self.status, self.reason = check.status, check.reason
        self.next_time = timestamp + 0.2
        return self.status

    def result(self):
        return {
            "schema_version": 1,
            "status": self.status,
            "reason": self.reason,
            "initial_result": self.initial.result(),
            "recheck_result": self.recheck.result() if self.recheck else None,
            "observation_count": self.observation_count,
        }
