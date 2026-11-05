"""Three fresh paired exposures with raw-evidence and motion guards."""

import copy
import math
from collections import deque
from statistics import median

from .cross_view import CrossViewConfirmation


class GuardedMedianPair:
    """A causal three-frame filter, never a certified uncertainty reduction.

    Missing/conflicting/outside raw evidence breaks the window. Motion is
    checked before smoothing, with both a rate and a window-span limit.
    """

    def __init__(self, position_error_m):
        validator = CrossViewConfirmation(0, position_error_m=position_error_m)
        if validator.position_error_m is None:
            raise ValueError("explicit position budget required")
        self.budget = validator.position_error_m
        self.history = deque(maxlen=3)
        self.last_time = None
        self.rows = []

    def observe(self, timestamp, primary, secondary):
        if (
            type(timestamp) not in (int, float)
            or not math.isfinite(timestamp)
            or (self.last_time is not None and timestamp <= self.last_time)
        ):
            raise ValueError("finite increasing time required")
        probe = CrossViewConfirmation(timestamp, position_error_m=self.budget)
        probe.step(timestamp, primary, secondary)
        observation = probe.rows[-1]
        reason = observation["gate"]
        output = (None, None)
        if reason != "paired_agreement":
            self.history.clear()
        else:
            points = (observation["primary"], observation["secondary"])
            if self.last_time is not None and timestamp - self.last_time > 0.25:
                self.history.clear()
            moving = bool(self.history) and any(
                math.dist(points[v], self.history[-1][1][v]) / (timestamp - self.history[-1][0])
                >= 0.02
                for v in range(2)
            )
            if moving:
                self.history.clear()
            self.history.append((timestamp, points))
            reason = "raw_motion" if moving else "collecting_three_frames"
            if len(self.history) == 3:
                span = max(
                    max(row[1][v][axis] for row in self.history)
                    - min(row[1][v][axis] for row in self.history)
                    for v in range(2)
                    for axis in range(3)
                )
                if span > 0.001:
                    self.history.clear()
                    self.history.append((timestamp, points))
                    reason = "window_motion_or_dispersion"
                else:
                    output = tuple(
                        tuple(median(row[1][v][axis] for row in self.history) for axis in range(3))
                        for v in range(2)
                    )
                    reason = "three_frame_median"
        self.last_time = timestamp
        self.rows.append(
            {
                "time": timestamp,
                "raw_gate": observation["gate"],
                "filter_gate": reason,
                "output": output,
                "window_size": len(self.history),
            }
        )
        return output

    def result(self):
        return copy.deepcopy(
            {
                "window_frames": 3,
                "max_gap_s": 0.25,
                "max_speed_m_s": 0.02,
                "max_axis_span_m": 0.001,
                "observations": self.rows,
            }
        )


class TemporalReferenceConfirmation:
    def __init__(self, start_time, *, position_error_m):
        self.confirmation = CrossViewConfirmation(start_time, position_error_m=position_error_m)
        self.filter = GuardedMedianPair(position_error_m)
        self.start_time = start_time

    @property
    def status(self):
        return self.confirmation.status

    def step(self, timestamp, primary, secondary, *, stop=False):
        if self.status != "pending":
            return self.status
        if stop:
            return self.confirmation.step(timestamp, None, None, stop=True)
        if not math.isfinite(timestamp) or timestamp < self.start_time:
            raise ValueError("invalid observation time")
        first, second = self.filter.observe(timestamp, primary, secondary)
        return self.confirmation.step(timestamp, first, second)

    def result(self):
        return {
            "status": self.status,
            "filter": self.filter.result(),
            "paired": self.confirmation.result(),
        }
