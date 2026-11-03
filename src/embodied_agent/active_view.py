"""Bounded view selection from detection failures, independent of object truth."""

import copy
import math


class ActiveView:
    """Switch once after two consecutive unusable primary observations.

    locate(view) must return a current 3D detection in the common world frame,
    using that view's calibration. Switching returns no observation so temporal
    evidence cannot span different views. This selects cameras, not arm motion.
    """

    def __init__(self):
        self.view = "primary"
        self.misses = 0
        self.last_time = None
        self.rows = []
        self.switches = []

    def observe(self, timestamp, locate):
        if (
            type(timestamp) not in (int, float)
            or not math.isfinite(timestamp)
            or (self.last_time is not None and timestamp <= self.last_time)
        ):
            raise ValueError("finite increasing observation time required")
        self.last_time = timestamp
        view = self.view
        error, point = None, None
        try:
            detected = locate(view)
            if detected is not None:
                point = tuple(float(x) for x in detected)
                if len(point) != 3 or not all(math.isfinite(x) for x in point):
                    raise ValueError("finite world-frame position required")
        except (ValueError, TypeError, OverflowError) as exc:
            point, error = None, str(exc)[:512]
        self.misses = self.misses + 1 if point is None else 0
        if view == "primary" and self.misses >= 2:
            self.view = "secondary"
            self.switches.append(
                {
                    "time": timestamp,
                    "from": view,
                    "to": self.view,
                    "reason": "two_consecutive_missing_detections",
                }
            )
            self.misses = 0
        self.rows.append(
            {
                "time": timestamp,
                "view": view,
                "xyz": list(point) if point is not None else None,
                "error": error,
            }
        )
        return point

    def result(self):
        return copy.deepcopy(
            {
                "schema_version": 1,
                "view": self.view,
                "switches": self.switches,
                "observations": self.rows,
            }
        )
