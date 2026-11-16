"""Bounded advance capture reservation; experimental, not a task controller."""

import math
import time
from dataclasses import dataclass

from .observation_capture import StampedPosition
from .threaded_pair_capture import CapturedPair


@dataclass(frozen=True)
class ScheduledPairRequest:
    request_id: int
    session_id: str
    host_time: float  # Actual reservation creation time, never rewritten as target.
    target_host_time: float
    latest_host_time: float

    def __post_init__(self):
        if type(self.request_id) is not int or not 1 <= self.request_id <= 36:
            raise ValueError("request id must be within 1..36")
        if not isinstance(self.session_id, str) or not 1 <= len(self.session_id) <= 256:
            raise ValueError("bounded nonempty session required")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (self.host_time, self.target_host_time, self.latest_host_time)
        ):
            raise ValueError("finite host timestamps required")
        if not 0 <= self.target_host_time - self.host_time <= 0.25:
            raise ValueError("reservation lead must be within 250 ms")
        if not 0 <= self.latest_host_time - self.target_host_time <= 0.01 + 1e-9:
            raise ValueError("exposure window must be within 10 ms")


def validate_spin_window(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 0.002
    ):
        raise ValueError("spin_window_s must be finite seconds in [0, .002]")


class ScheduledPairReader:
    """Wait for exposure, optionally spin near target, then validate real timestamps.

    Own this adapter in a cancellable worker process. Waiting does not offer
    real-time guarantees; reader blocking is handled by the process owner.
    No retries, timestamp substitution, or task-success promotion are performed.
    """

    def __init__(self, reader, *, clock=time.monotonic, sleep=time.sleep, spin_window_s=0):
        if not all(callable(v) for v in (reader, clock, sleep)):
            raise ValueError("reader, clock and sleep must be callable")
        validate_spin_window(spin_window_s)
        self.spin_window_s = spin_window_s
        self.reader, self.clock, self.sleep = reader, clock, sleep

    def __call__(self, request):
        if not isinstance(request, ScheduledPairRequest):
            raise TypeError("ScheduledPairRequest required")
        previous = request.host_time

        def now():
            nonlocal previous
            value = self.clock()
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < previous
            ):
                raise ValueError("scheduled_capture_clock_invalid")
            previous = value
            return value

        current = now()
        while current < request.target_host_time:
            remaining = request.target_host_time - current
            if remaining > self.spin_window_s:
                self.sleep(min(remaining - self.spin_window_s, 0.001))
            current = now()
        if current > request.latest_host_time:
            raise ValueError("scheduled_capture_window_missed")
        pair = self.reader(request)
        returned = now()
        if not isinstance(pair, CapturedPair) or pair.session_id != request.session_id:
            raise ValueError("scheduled_capture_session_or_pair_invalid")
        for point in (pair.primary, pair.secondary):
            if (
                not isinstance(point, StampedPosition)
                or isinstance(point.captured_at, bool)
                or not isinstance(point.captured_at, (int, float))
                or not math.isfinite(point.captured_at)
                or not request.target_host_time <= point.captured_at <= request.latest_host_time
                or not current <= point.captured_at <= returned
            ):
                raise ValueError("scheduled_capture_exposure_outside_window")
        return pair


@dataclass(frozen=True)
class ScheduledReaderFactory:
    """Spawn-picklable adapter; the underlying reader is created in its worker."""

    factory: object
    spin_window_s: float = 0

    def __call__(self):
        return ScheduledPairReader(self.factory(), spin_window_s=self.spin_window_s)
