"""Frozen per-action calibration with sticky conservative fallback."""

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
    def __init__(self, samples, *, fallback_s=0.05, margin_s=0.005, minimum_samples=20):
        self.fallback_s = seconds(fallback_s)
        self.margin_s = seconds(margin_s)
        if type(minimum_samples) is not int or minimum_samples < 2:
            raise ValueError("minimum_samples must be at least two")
        self.estimates = {}
        self.degraded = set()
        for action, values in samples.items():
            if not isinstance(action, str) or not action:
                raise ValueError("action name required")
            values = tuple(seconds(value) for value in values)
            if len(values) >= minimum_samples:
                self.estimates[action] = max(values) + self.margin_s

    def required(self, action):
        estimate = self.estimates.get(action, self.fallback_s)
        return max(self.fallback_s, estimate) if action in self.degraded else estimate

    def record(self, action, elapsed_s, *, confirmed):
        if not confirmed or elapsed_s > self.required(action):
            self.degraded.add(action)
        # Validation data never refits or lowers the frozen estimates.
