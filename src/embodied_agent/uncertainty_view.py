"""Choose one fresh paired view configuration from boundary uncertainty."""

import copy
import math

from .cross_view import CrossViewConfirmation


class UncertaintyViewConfirmation:
    """Bounded paired acquisition policy, with externally supplied error bounds.

    locate(profile) returns two synchronous corrected world-frame positions.
    A profile selects a *pair* of calibrated views, not a replacement error bar
    for old observations. Caller must actually acquire that configuration.
    No robot/camera motion or calibration is performed here.
    """

    def __init__(self, start_time, profiles, *, initial="coarse", switch_on_missing=False):
        if not isinstance(profiles, dict) or initial not in profiles:
            raise ValueError("named initial acquisition profile required")
        self.profiles = {}
        for name, bounds in profiles.items():
            if not isinstance(name, str) or not name or bounds is None:
                raise ValueError("named profiles with explicit error bounds required")
            validator = CrossViewConfirmation(start_time, position_error_m=bounds)
            self.profiles[name] = validator.position_error_m
        self.start = float(start_time)
        self.profile = initial
        self.confirmation = self._new_confirmation()
        self.last_time = None
        self.rejections = 0
        if type(switch_on_missing) is not bool:
            raise ValueError("switch_on_missing must be boolean")
        self.switch_on_missing = switch_on_missing
        self.misses = 0
        self.switches = []
        self.rows = []

    def _new_confirmation(self):
        return CrossViewConfirmation(self.start, position_error_m=self.profiles[self.profile])

    @property
    def status(self):
        return self.confirmation.status

    @staticmethod
    def _margin(points, bounds):
        return min(
            min(
                0.05 - abs(p[0] - 0.64) - bounds[0],
                0.04 - abs(p[1]) - bounds[1],
                p[2] - 0.032 - bounds[2],
                0.058 - p[2] - bounds[2],
            )
            for p in points
        )

    def step(self, timestamp, locate, *, stop=False):
        if self.status != "pending":
            return self.status
        if stop:
            return self.confirmation.step(timestamp, None, None, stop=True)
        if (
            type(timestamp) not in (int, float)
            or not math.isfinite(timestamp)
            or timestamp < self.start
            or (self.last_time is not None and timestamp <= self.last_time)
        ):
            raise ValueError("finite increasing observation time required")
        # The scheduler owns the deadline, including after a profile change.
        # Do not start another acquisition at/after the original deadline.
        expired = timestamp - self.start >= 3
        profile, error = self.profile, None
        primary, secondary = None, None
        if not expired:
            try:
                primary, secondary = locate(profile)
            except (ValueError, TypeError, OverflowError) as exc:
                error = str(exc)[:512]
        self.confirmation.step(timestamp, primary, secondary)
        self.last_time = timestamp
        observation = self.confirmation.rows[-1]
        gate = observation["gate"]
        self.rejections = self.rejections + 1 if gate == "uncertainty_overlaps_boundary" else 0
        self.misses = self.misses + 1 if gate == "missing_or_invalid_view" else 0
        candidates = {}
        chosen, switch_reason = None, None
        if self.status == "pending" and not self.switches and timestamp - self.start <= 1.8:
            if self.rejections >= 2:
                points = (observation["primary"], observation["secondary"])
                candidates = {
                    name: self._margin(points, bounds)
                    for name, bounds in self.profiles.items()
                    if name != profile
                }
                # Positive predicted containment only justifies acquiring new evidence.
                eligible = [name for name, margin in candidates.items() if margin > 1e-12]
                if eligible:
                    chosen = min(eligible, key=lambda name: (-candidates[name], name))
                    switch_reason = "boundary_uncertainty"
            elif self.switch_on_missing and self.misses >= 2:
                eligible = [name for name in self.profiles if name != profile]
                if eligible:
                    # No position is available to predict containment. Prefer the
                    # smallest supplied error-box diagonal as a bounded fallback.
                    chosen = min(
                        eligible, key=lambda name: (math.hypot(*self.profiles[name]), name)
                    )
                    switch_reason = "two_missing_pairs"
            if chosen is not None:
                self.profile = chosen
                self.switches.append(
                    {"time": timestamp, "from": profile, "to": chosen, "reason": switch_reason}
                )
                self.confirmation = self._new_confirmation()
                self.confirmation.step(timestamp, None, None)
                self.rejections = self.misses = 0
        self.rows.append(
            {
                "time": timestamp,
                "profile": profile,
                "error": error,
                "deadline_expired": expired,
                "observation": copy.deepcopy(observation),
                "predicted_candidate_margins_m": candidates,
                "next_profile": self.profile,
            }
        )
        return self.status

    def result(self):
        return copy.deepcopy(
            {
                "schema_version": 1,
                "status": self.status,
                "profile": self.profile,
                "profiles": self.profiles,
                "switch_on_missing": self.switch_on_missing,
                "switches": self.switches,
                "observations": self.rows,
                "paired": self.confirmation.result(),
            }
        )
