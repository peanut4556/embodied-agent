"""Withhold a candidate confirmation until a bounded high-rate visual challenge."""

import copy
import math

from .cross_view import CrossViewConfirmation
from .settled_reference import SettledReferenceConfirmation


class ConfirmationChallenge:
    """Request 5 Hz until a candidate, then one second at 20 Hz.

    Caller must schedule fresh synchronized paired captures at next_capture_time.
    No hidden motion truth or device control is used. Rejected challenges are
    terminal; the configured deadline and 31-capture cap never grow during a run.
    The default capture budget is zero; experiments may explicitly reserve time.
    """

    def __init__(self, start_time, *, position_error_m, capture_budget_s=0.0):
        if (
            isinstance(capture_budget_s, bool)
            or not isinstance(capture_budget_s, (int, float))
            or not math.isfinite(capture_budget_s)
            or not 0 <= capture_budget_s <= 0.25
        ):
            raise ValueError("capture budget must be finite and between 0 and 0.25 seconds")
        self.capture_budget_s = float(capture_budget_s)
        self.baseline = SettledReferenceConfirmation(start_time, position_error_m=position_error_m)
        self.budget = self.baseline.confirmation.position_error_m
        self.start, self.deadline = float(start_time), float(start_time) + 3 + self.capture_budget_s
        self.status, self.phase, self.reason = "pending", "search", "collecting_candidate"
        self.next_capture_time = float(start_time)
        self.last_time = self.candidate_time = None
        self.points, self.rows = [], []

    def step(self, timestamp, primary, secondary, *, stop=False):
        if self.status != "pending":
            return self.status
        if stop:
            self.status, self.reason = "cancelled", "user_stop"
            self.next_capture_time = None
            return self.status
        if (
            not math.isfinite(timestamp)
            or timestamp < self.start
            or (self.last_time is not None and timestamp <= self.last_time)
            or timestamp < self.next_capture_time - 1e-9
        ):
            raise ValueError("fresh scheduled capture time required")
        previous = self.last_time
        self.last_time = timestamp
        span = None
        if timestamp > self.deadline + 1e-9:
            self.status, self.reason = "unconfirmed", "deadline_exceeded"
        elif self.phase == "search":
            self.baseline.step(timestamp, primary, secondary)
            if self.baseline.status == "confirmed_in_tray":
                self.candidate_time = timestamp
                if timestamp + 1 > self.deadline + 1e-9:
                    self.status, self.reason = "unconfirmed", "insufficient_challenge_time"
                else:
                    self.phase, self.reason = "challenge", "candidate_withheld"
            elif self.baseline.status != "pending":
                self.status, self.reason = "unconfirmed", "no_candidate"
        if self.status == "pending" and self.phase == "challenge":
            if self.points and timestamp - previous > 0.06 + 1e-9:
                self.status, self.reason = "unconfirmed", "challenge_capture_gap"
            else:
                probe = CrossViewConfirmation(timestamp, position_error_m=self.budget)
                probe.step(timestamp, primary, secondary)
                row = probe.rows[-1]
                if row["gate"] != "paired_agreement":
                    self.status, self.reason = "unconfirmed", "challenge_view_rejected"
                else:
                    self.points.append((row["primary"], row["secondary"]))
                    span = max(
                        max(p[v][a] for p in self.points) - min(p[v][a] for p in self.points)
                        for v in range(2)
                        for a in range(3)
                    )
                    if span > 0.001 + 1e-12:
                        self.status, self.reason = "unconfirmed", "challenge_motion"
                    elif timestamp - self.candidate_time >= 1 - 1e-9 and len(self.points) >= 21:
                        self.status, self.reason = "confirmed_in_tray", "challenge_passed"
        self.rows.append(
            {
                "time": timestamp,
                "phase": self.phase,
                "status": self.status,
                "reason": self.reason,
                "raw_axis_span_m": span,
            }
        )
        if self.status == "pending" and (timestamp >= self.deadline - 1e-9 or len(self.rows) >= 31):
            self.status, self.reason = "unconfirmed", "challenge_budget_exhausted"
            self.rows[-1].update(status=self.status, reason=self.reason)
        self.next_capture_time = (
            min(self.deadline, timestamp + (0.05 if self.phase == "challenge" else 0.2))
            if self.status == "pending"
            else None
        )
        return self.status

    def result(self):
        return copy.deepcopy(
            {
                "status": self.status,
                "phase": self.phase,
                "reason": self.reason,
                "candidate_time": self.candidate_time,
                "next_capture_time": self.next_capture_time,
                "captures": len(self.rows),
                "capture_limit": 31,
                "capture_budget_s": self.capture_budget_s,
                "deadline": self.deadline,
                "observations": self.rows,
                "baseline": self.baseline.result(),
            }
        )
