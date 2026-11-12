"""Nonblocking caller loop joining the two-stage state machine to real workers."""

import time

from .threaded_pair_capture import ThreadedPairCapture
from .two_stage_capture import TwoStageCaptureController


class TwoStageCaptureRuntime:
    """Caller repeatedly invokes step; no autonomous task or motion is started.

    The supplied clock is the same host monotonic clock used by the camera.
    Drain before poll, accounting for queue age at actual controller consumption.
    step returns only status; expensive audit snapshots are explicit via result().
    """

    def __init__(self, controller, read, *, clock=time.monotonic):
        if not isinstance(controller, TwoStageCaptureController):
            raise TypeError("TwoStageCaptureController required")
        if controller.requests or controller.status != "pending":
            raise ValueError("unused pending controller required")
        self.controller = controller
        self.capture = ThreadedPairCapture(read)
        self.clock = clock
        self.errors = []

    def _now(self):
        host = self.clock()
        m = self.controller.mapping
        return m.task_origin + (host - m.host_origin) * m.rate, host

    def step(self, *, stop=False):
        c = self.controller
        if c.status != "pending":
            self.capture.cancel()
            return c.status
        if stop:
            # Stop must remain available even when the external clock fails.
            c.poll(c.candidate.check.start, c.mapping.host_origin, stop=True)
            self.capture.cancel()
            return c.status
        try:
            for request_id, pair, error in self.capture.drain():
                if c.status != "pending":
                    break
                now, host = self._now()
                if error:
                    self.errors.append({"request_id": request_id, "error": error})
                c.complete(
                    request_id,
                    now,
                    host,
                    pair.primary if pair and not error else None,
                    pair.secondary if pair and not error else None,
                    session_id=pair.session_id if pair and not error else "",
                )
            if c.status == "pending":
                now, host = self._now()
                req = c.poll(now, host)
                if req is not None and not self.capture.submit(req):
                    c.driver_failed("capture_capacity_unavailable")
        except Exception:  # Abort ownership before propagating caller/clock errors.
            c.driver_failed("capture_runtime_exception")
            self.capture.cancel()
            raise
        if c.status != "pending":
            self.capture.cancel()
        return c.status

    def close(self, *, wait_s=0.0):
        if self.controller.status == "pending":
            self.controller.driver_failed("runtime_closed")
        self.capture.cancel()
        return self.capture.reap(wait_s)

    def result(self):
        return {
            "confirmation": self.controller.result(),
            "capture": self.capture.result(),
            "driver_errors": [dict(e) for e in self.errors],
        }
