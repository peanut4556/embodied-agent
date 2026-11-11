"""Optional post-candidate gate; requires a separate low-latency observation path."""

import copy
import math

from .cross_view import CrossViewConfirmation
from .observation_capture import StampedPosition
from .phase_sweep_challenge import PhaseSweepChallengeController


class FreshWitnessGate:
    """Five new paired exposures, >=40 ms span, <=20 ms delivery age.

    Candidate remains historical evidence. No driver, request retry, or claim of
    future stability is provided. All methods run on one controller thread.
    """

    def __init__(self, candidate):
        if (
            not isinstance(candidate, PhaseSweepChallengeController)
            or candidate.status != "confirmed_in_tray"
        ):
            raise ValueError("completed phase-sweep candidate required")
        self.candidate = candidate.result()
        self.start = candidate.events[-1]["delivery_time"]
        self.deadline = self.start + 0.1
        self.mapping = candidate.check.clock_map
        self.session = candidate.check.session
        self.budget = candidate.check.evidence.budget
        self.anchor = copy.deepcopy(candidate.check.evidence.points[-1])
        self.points = [self.anchor]
        self.status, self.reason = "pending", "awaiting_fresh_witness"
        self.last_task = self.last_exposure = self.first_exposure = None
        self.rows = []

    def _time(self, now):
        if (
            not math.isfinite(now)
            or now < self.start
            or (self.last_task is not None and now < self.last_task)
        ):
            raise ValueError("monotonic finite witness task time required")

    def _end(self, reason, status="unconfirmed"):
        self.status, self.reason = status, reason
        return self.status

    def tick(self, task_now, *, stop=False):
        if self.status != "pending":
            return self.status
        if stop:
            return self._end("user_stop", "cancelled")
        self._time(task_now)
        self.last_task = task_now
        if task_now >= self.deadline - 1e-9:
            return self._end("witness_deadline")
        return self.status

    def deliver(self, task_now, host_now, primary, secondary, *, session_id):
        if self.status != "pending":
            return self.status
        self._time(task_now)
        if self.rows and task_now <= self.rows[-1]["delivery_time"]:
            raise ValueError("strictly increasing witness delivery required")
        self.last_task = task_now
        exposure = age = None
        error = None
        try:
            if task_now > self.deadline + 1e-9:
                raise ValueError("witness_deadline")
            if session_id != self.session:
                raise ValueError("camera_session_changed")
            if any(not isinstance(p, StampedPosition) for p in (primary, secondary)):
                raise ValueError("missing_timestamped_pair")
            times = [
                self.mapping.map(p.captured_at, host_now, task_now) for p in (primary, secondary)
            ]
            if abs(times[0] - times[1]) > 1e-9:
                raise ValueError("unsynchronized_witness")
            exposure = min(times)
            age = task_now - exposure
            if exposure < self.start - 1e-9:
                raise ValueError("pre_candidate_exposure")
            if age > 0.02 + 1e-9:
                raise ValueError("stale_witness")
            if (
                self.last_exposure is not None
                and not 0.01 - 1e-9 <= exposure - self.last_exposure <= 0.03 + 1e-9
            ):
                raise ValueError("witness_capture_gap")
            probe = CrossViewConfirmation(exposure, position_error_m=self.budget)
            probe.step(exposure, primary.position, secondary.position)
            row = probe.rows[-1]
            if row["gate"] != "paired_agreement":
                raise ValueError("witness_view_rejected")
            self.points.append((row["primary"], row["secondary"]))
            span = max(
                max(p[v][a] for p in self.points) - min(p[v][a] for p in self.points)
                for v in range(2)
                for a in range(3)
            )
            if span > 0.001 + 1e-12:
                raise ValueError("witness_motion")
            if self.first_exposure is None:
                self.first_exposure = exposure
            self.last_exposure = exposure
            if len(self.points) == 6:
                if exposure - self.first_exposure < 0.04 - 1e-9:
                    raise ValueError("insufficient_witness_span")
                self._end("fresh_witness_passed", "confirmed_recent_evidence")
        except (ValueError, TypeError, OverflowError) as exc:
            error = str(exc)
            self._end(error)
        self.rows.append(
            {
                "delivery_time": task_now,
                "exposure_time": exposure,
                "age_s": age,
                "status": self.status,
                "reason": self.reason,
                "error": error,
            }
        )
        if self.status == "pending" and task_now >= self.deadline - 1e-9:
            self._end("witness_deadline")
        return self.status

    def result(self):
        return copy.deepcopy(
            {
                "status": self.status,
                "reason": self.reason,
                "candidate": self.candidate,
                "start": self.start,
                "deadline": self.deadline,
                "witness_limit": 5,
                "max_delivery_age_s": 0.02,
                "first_exposure_s": self.first_exposure,
                "last_exposure_s": self.last_exposure,
                "decision_time_s": self.rows[-1]["delivery_time"] if self.rows else None,
                "guarantees_current_or_future_stability": False,
                "observations": self.rows,
            }
        )
