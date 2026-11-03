"""Bounded translation correction from an external fixed reference catalog."""

import copy
import math

from .cross_view import CrossViewConfirmation


def point3(value):
    point = tuple(float(x) for x in value)
    if len(point) != 3 or not all(math.isfinite(x) for x in point):
        raise ValueError("finite 3D position required")
    return point


class ReferenceConfirmation:
    """Three identified references must accompany each target capture.

    Corrects translation only, never fits a target position or updates the
    supplied reference catalog. Correct catalog and capture alignment are caller
    obligations; a common error in that catalog can defeat this guard.
    """

    def __init__(self, start_time, reference_points):
        if not isinstance(reference_points, dict) or len(reference_points) != 3:
            raise ValueError("three named reference positions required")
        if not all(isinstance(key, str) and key for key in reference_points):
            raise ValueError("named references required")
        self.reference_points = {key: point3(value) for key, value in reference_points.items()}
        a, b, c = self.reference_points.values()
        ab, ac = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
        area = math.hypot(
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        if not math.isfinite(area) or area <= 1e-6:
            raise ValueError("spatially separated noncollinear references required")
        self.confirmation = CrossViewConfirmation(start_time)
        self.rows = []

    @property
    def status(self):
        return self.confirmation.status

    def correct(self, target, references):
        try:
            point = point3(target)
            if not isinstance(references, dict) or set(references) != set(self.reference_points):
                raise ValueError("complete identified references required")
            residuals = []
            for key, expected in self.reference_points.items():
                observed = point3(references[key])
                residuals.append(tuple(observed[i] - expected[i] for i in range(3)))
            shift = tuple(math.fsum(row[i] / 3 for row in residuals) for i in range(3))
            spread = max(math.dist(row, shift) for row in residuals)
            if not math.isfinite(spread) or spread > 0.003:
                raise ValueError("reference residuals inconsistent")
            if math.hypot(*shift) > 0.03:
                raise ValueError("reference translation exceeds budget")
            corrected = point3(tuple(point[i] - shift[i] for i in range(3)))
            return corrected, {"accepted": True, "translation": shift, "max_residual": spread}
        except (ValueError, TypeError, OverflowError) as exc:
            return None, {"accepted": False, "error": str(exc)[:512]}

    def step(
        self, timestamp, primary, secondary, primary_references, secondary_references, *, stop=False
    ):
        if self.status != "pending":
            return self.status
        if stop:
            return self.confirmation.step(timestamp, None, None, stop=True)
        first, first_check = self.correct(primary, primary_references)
        second, second_check = self.correct(secondary, secondary_references)
        self.confirmation.step(timestamp, first, second)
        self.rows.append(
            {
                "time": timestamp,
                "primary": first_check,
                "secondary": second_check,
                "corrected_primary": first,
                "corrected_secondary": second,
            }
        )
        return self.status

    def result(self):
        return copy.deepcopy(
            {
                "schema_version": 1,
                "status": self.status,
                "reference_points": self.reference_points,
                "translation_limit_m": 0.03,
                "residual_limit_m": 0.003,
                "corrections": self.rows,
                "paired": self.confirmation.result(),
            }
        )
