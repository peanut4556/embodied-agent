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

    def __init__(self, start_time):
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
        if all(point is not None for point in points):
            distance = math.dist(*points)
            if not math.isfinite(distance) or distance > 0.01:
                reason = "position_disagreement"
            elif _inside(points[0]) != _inside(points[1]):
                reason = "region_disagreement"
            else:
                reason, accepted = "paired_agreement", points[0]
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
                "observations": self.rows,
            }
        )
