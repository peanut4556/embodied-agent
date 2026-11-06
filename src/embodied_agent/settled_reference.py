"""Longer raw-observation stationarity gate ahead of temporal confirmation."""

import copy
import math
from collections import deque

from .cross_view import CrossViewConfirmation
from .reference_confirmation import point3
from .temporal_reference import GuardedMedianPair


class SettledReferenceConfirmation:
    """Require >=1 s raw range evidence before accumulating confirmation.

    Each view/axis must stay within 1 mm over the retained window. This is an
    observed range limit, not a proof of zero true velocity or future safety.
    """

    def __init__(self, start_time, *, position_error_m):
        self.confirmation = CrossViewConfirmation(start_time, position_error_m=position_error_m)
        self.filter = GuardedMedianPair(position_error_m)
        self.start = start_time
        self.history = deque()
        self.last_time = None
        self.rows = []

    @property
    def status(self):
        return self.confirmation.status

    def step(self, timestamp, primary, secondary, *, stop=False):
        if self.status != "pending":
            return self.status
        if stop:
            return self.confirmation.step(timestamp, None, None, stop=True)
        if not math.isfinite(timestamp) or timestamp < self.start:
            raise ValueError("invalid observation time")
        filtered = self.filter.observe(timestamp, primary, secondary)
        raw = self.filter.rows[-1]
        if self.last_time is not None and timestamp - self.last_time > 0.25:
            self.history.clear()
        self.last_time = timestamp
        reason, span = "insufficient_raw_duration", None
        if raw["raw_gate"] != "paired_agreement":
            self.history.clear()
            reason = "raw_evidence_rejected"
        else:
            if raw["filter_gate"] == "raw_motion":
                self.history.clear()
            self.history.append((timestamp, (point3(primary), point3(secondary))))
            # Keep the closest sample at/before t-1s plus all newer samples.
            while len(self.history) > 1 and timestamp - self.history[1][0] >= 1 - 1e-9:
                self.history.popleft()
            if timestamp - self.history[0][0] >= 1 - 1e-9:
                span = max(
                    max(r[1][v][a] for r in self.history) - min(r[1][v][a] for r in self.history)
                    for v in range(2)
                    for a in range(3)
                )
                reason = "raw_range_stable" if span <= 0.001 + 1e-12 else "sustained_raw_drift"
        accepted = filtered if reason == "raw_range_stable" else (None, None)
        self.confirmation.step(timestamp, *accepted)
        self.rows.append(
            {
                "time": timestamp,
                "gate": reason,
                "raw_axis_span_m": span,
                "window_frames": len(self.history),
                "status": self.status,
            }
        )
        return self.status

    def result(self):
        return copy.deepcopy(
            {
                "status": self.status,
                "raw_window_s": 1,
                "max_raw_axis_span_m": 0.001,
                "observations": self.rows,
                "filter": self.filter.result(),
                "paired": self.confirmation.result(),
            }
        )
