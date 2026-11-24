"""Frozen per-action calibration with sticky conservative fallback."""

import copy
import hashlib
import json
import math


def seconds(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("durations must be finite positive seconds")
    return value


class ActionBudget:
    def __init__(
        self,
        samples,
        *,
        fallback_s=0.05,
        margin_s=0.005,
        minimum_samples=20,
        drift_policy="fallback",
    ):
        if drift_policy not in ("fallback", "quarantine"):
            raise ValueError("drift_policy must be fallback or quarantine")
        self.drift_policy = drift_policy
        self.fallback_s = seconds(fallback_s)
        self.margin_s = seconds(margin_s)
        if type(minimum_samples) is not int or minimum_samples < 2:
            raise ValueError("minimum_samples must be at least two")
        self.minimum_samples = minimum_samples
        self.revision = 0
        self._recalibrations = []
        self.estimates = {}
        self.degraded = set()
        for action, values in samples.items():
            if not isinstance(action, str) or not action:
                raise ValueError("action name required")
            values = tuple(seconds(value) for value in values)
            if len(values) >= minimum_samples:
                self.estimates[action] = max(values) + self.margin_s

    def recalibrated(self, action, samples, *, batch_id):
        """Return a new budget; restore only this degraded action after validation."""
        if action not in self.degraded:
            raise ValueError("only degraded actions may be recalibrated")
        if (
            not isinstance(batch_id, str)
            or not batch_id.strip()
            or batch_id != batch_id.strip()
            or len(batch_id) > 128
        ):
            raise ValueError("batch_id must be nonempty and at most 128 characters")
        if any(
            batch_id in (row["batch_id"], row.get("validation_batch_id"))
            for row in self._recalibrations
        ):
            raise ValueError("calibration batch already used")
        values = tuple(seconds(value) for value in samples)
        if len(values) < self.minimum_samples:
            raise ValueError("insufficient recalibration samples")
        estimate = max(values) + self.margin_s
        if not math.isfinite(estimate):
            raise ValueError("calibration estimate overflow")
        replacement = copy.deepcopy(self)
        replacement.revision += 1
        replacement.estimates[action] = estimate
        replacement.degraded.remove(action)
        replacement._recalibrations.append(
            {
                "revision": replacement.revision,
                "action": action,
                "batch_id": batch_id,
                "sample_count": len(values),
                "previous_required_s": self.required(action),
                "required_s": estimate,
                "samples_sha256": hashlib.sha256(
                    json.dumps(values, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
        return replacement

    def validated_recalibrated(
        self, action, samples, *, batch_id, validation_samples, validation_batch_id
    ):
        """Gate a frozen candidate on a separate batch; failures leave self untouched."""
        if (
            not isinstance(validation_batch_id, str)
            or not validation_batch_id.strip()
            or validation_batch_id != validation_batch_id.strip()
            or len(validation_batch_id) > 128
            or validation_batch_id == batch_id
        ):
            raise ValueError("distinct nonempty validation batch required")
        if any(
            validation_batch_id in (row["batch_id"], row.get("validation_batch_id"))
            for row in self._recalibrations
        ):
            raise ValueError("validation batch already used")
        candidate = self.recalibrated(action, samples, batch_id=batch_id)
        validation = tuple(seconds(value) for value in validation_samples)
        if len(validation) < self.minimum_samples:
            raise ValueError("insufficient validation samples")
        if max(validation) > candidate.required(action):
            raise ValueError("validation exceeds calibrated budget; action remains degraded")
        candidate._recalibrations[-1].update(
            validation_batch_id=validation_batch_id,
            validation_sample_count=len(validation),
            validation_max_s=max(validation),
            validation_sha256=hashlib.sha256(
                json.dumps(validation, separators=(",", ":")).encode()
            ).hexdigest(),
        )
        return candidate

    def recalibration_history(self):
        return copy.deepcopy(self._recalibrations)

    def blocked(self, action):
        return action != "stop" and self.drift_policy == "quarantine" and action in self.degraded

    def required(self, action):
        estimate = self.estimates.get(action, self.fallback_s)
        return max(self.fallback_s, estimate) if action in self.degraded else estimate

    def record(self, action, elapsed_s, *, confirmed):
        if (
            type(confirmed) is not bool
            or isinstance(elapsed_s, bool)
            or not isinstance(elapsed_s, (int, float))
            or not math.isfinite(elapsed_s)
            or elapsed_s < 0
        ):
            raise ValueError("valid elapsed seconds and boolean confirmation required")
        if not confirmed or elapsed_s > self.required(action):
            self.degraded.add(action)
        # Validation data never refits or lowers the frozen estimates.
