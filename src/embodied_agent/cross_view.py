"""Reject inconsistent paired world-frame detections before temporal confirmation."""

import copy
import math

from .release_confirmation import ReleaseConfirmation


def _inside(point):
    return abs(point[0] - 0.64) < 0.05 and abs(point[1]) < 0.04 and 0.032 < point[2] < 0.058


class CrossViewConfirmation:
    """Synchronous paired detections only; missing/conflicting views fail closed.

    Caller must guarantee both positions correspond to the same capture epoch
    and use independently calibrated world coordinates. Agreement is not truth.
    """

    def __init__(self, start_time, *, position_error_m=None):
        # Bounds apply to corrected world coordinates, including shared errors.
        # They are supplied assumptions, never inferred from view agreement.
        self.position_error_m = None
        if position_error_m is not None:
            try:
                bound = tuple(float(x) for x in position_error_m)
                if len(bound) != 3 or not all(math.isfinite(x) and x >= 0 for x in bound):
                    raise ValueError
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("three finite nonnegative position error bounds required") from exc
            self.position_error_m = bound
        self.confirmation = ReleaseConfirmation(start_time)
        self.status = "pending"
        self.reason = "collecting_evidence"
        self.rows = []

    def step(self, timestamp, primary, secondary, *, stop=False):
        if self.status != "pending":
            return self.status
        if stop:
            self.status, self.reason = "cancelled", "user_stop"
            return self.status
        points, errors = [], []
        for value in (primary, secondary):
            try:
                if value is None:
                    raise ValueError("missing detection")
                point = tuple(float(x) for x in value)
                if len(point) != 3 or not all(math.isfinite(x) for x in point):
                    raise ValueError("finite 3D detection required")
                points.append(point)
                errors.append(None)
            except (TypeError, ValueError, OverflowError) as exc:
                points.append(None)
                errors.append(str(exc)[:512])
        distance = None
        reason = "missing_or_invalid_view"
        accepted = None
        margins = None
        if all(point is not None for point in points):
            distance = math.dist(*points)
            if not math.isfinite(distance) or distance > 0.01:
                reason = "position_disagreement"
            elif _inside(points[0]) != _inside(points[1]):
                reason = "region_disagreement"
            else:
                reason, accepted = "paired_agreement", points[0]
                if self.position_error_m is not None:
                    # Union containment: both view boxes must fit. Intersecting
                    # them would silently assume independent, valid errors.
                    margins = [
                        min(
                            0.05 - abs(p[0] - 0.64) - self.position_error_m[0],
                            0.04 - abs(p[1]) - self.position_error_m[1],
                            p[2] - 0.032 - self.position_error_m[2],
                            0.058 - p[2] - self.position_error_m[2],
                        )
                        for p in points
                    ]
                    if min(margins) <= 1e-12:
                        reason, accepted = "uncertainty_overlaps_boundary", None
        # Preserve the original 3 s budget and break continuity on every rejection.
        self.confirmation.observe(timestamp, accepted)
        self.status, self.reason = self.confirmation.status, self.confirmation.reason
        self.rows.append(
            {
                "time": timestamp,
                "primary": points[0],
                "secondary": points[1],
                "errors": errors,
                "distance": distance if distance is None or math.isfinite(distance) else None,
                "gate": reason,
                "containment_margins_m": margins,
                "accepted": accepted,
                "status": self.status,
            }
        )
        return self.status

    def result(self):
        return copy.deepcopy(
            {
                "schema_version": 1,
                "status": self.status,
                "reason": self.reason,
                "max_disagreement_m": 0.01,
                "position_error_m": self.position_error_m,
                "observations": self.rows,
            }
        )
