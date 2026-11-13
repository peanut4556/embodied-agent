"""Explicit, bounded recovery with fresh task/session state after verified worker exit."""

import copy
import time
import uuid

from .exposure_confirmation import ExposureClockMap
from .process_pair_capture import ProcessPairCapture
from .two_stage_capture import TwoStageCaptureController
from .two_stage_runtime import TwoStageCaptureRuntime


class CaptureRecovery:
    """Single owner thread; start is explicit, never an automatic task replay.

    At most three generations. Old workers must be reaped before another factory
    is invoked. This only proves process isolation, not device recovery readiness.
    """

    def __init__(self, *, position_error_m, max_generations=3):
        if type(max_generations) is not int or not 1 <= max_generations <= 3:
            raise ValueError("one to three generations required")
        self.budget = tuple(position_error_m)
        self.limit = max_generations
        self.attempts = 0
        self.runtime = self.backend = None
        self.session = None
        self.state = "idle"
        self.history = []
        self.ignored_sessions = 0
        self._recorded = False

    def _retire(self):
        if self.backend is None:
            return True
        if self.runtime is not None:
            cleanup = self.runtime.close(wait_s=0.25)
        else:
            self.backend.cancel()
            cleanup = self.backend.reap(0.25)
        verified = (
            cleanup["closed"]
            and cleanup["workers_stopped"]
            and all(w["exitcode"] is not None for w in cleanup["workers"])
        )
        if not verified:
            self.state = "cleanup_pending"
            return False
        if not self._recorded:
            self.history.append(
                {
                    "session_id": self.session,
                    "generation": self.attempts,
                    "result": self.runtime.result() if self.runtime else None,
                    "cleanup": cleanup,
                }
            )
            self._recorded = True
        return True

    def start(self, factory, *, startup_timeout=5.0):
        """Return a new opaque session token, or None on startup/cleanup failure."""
        if self.state == "ownership_uncertain":
            raise RuntimeError("spawn ownership unresolved; recovery prohibited")
        if self.runtime is not None and self.runtime.controller.status == "pending":
            raise RuntimeError("stop the active task explicitly before recovery")
        if self.attempts >= self.limit:
            raise RuntimeError("recovery generation budget exhausted")
        if not self._retire():
            return None
        self.attempts += 1
        self.session = uuid.uuid4().hex
        self.runtime = self.backend = None
        self._recorded = False
        self.state = "starting"
        try:
            self.backend = ProcessPairCapture(factory)
        except Exception:
            # Constructor failure may hide partially-created OS resources. Never
            # guess that a subsequent start is safe merely because spawn raised.
            self.state = "ownership_uncertain"
            raise
        try:
            if not self.backend.wait_ready(startup_timeout):
                self.state = "startup_failed"
                self._retire()
                return None
            origin = time.monotonic()
            controller = TwoStageCaptureController(
                0,
                ExposureClockMap(origin, 0),
                self.session,
                position_error_m=self.budget,
            )
            self.runtime = TwoStageCaptureRuntime(controller, capture_backend=self.backend)
            self.state = "running"
            return self.session
        except Exception:
            self.state = "startup_failed"
            self._retire()
            raise

    def step(self, session_id, *, stop=False):
        if session_id != self.session or self.runtime is None:
            self.ignored_sessions += 1
            return None
        try:
            return self.runtime.step(stop=stop)
        finally:
            if self.runtime.controller.status != "pending":
                self.state = "terminal"

    def close(self):
        """Explicitly close an active task; retry if cleanup remains pending."""
        if self.state == "ownership_uncertain":
            return False
        if self._retire():
            self.state = "closed"
            return True
        return False

    def result(self):
        return copy.deepcopy(
            {
                "state": self.state,
                "session_id": self.session,
                "attempts": self.attempts,
                "generation_limit": self.limit,
                "ignored_sessions": self.ignored_sessions,
                "history": self.history,
                "runtime": self.runtime.result() if self.runtime else None,
                "hardware_recovery_verified": False,
            }
        )
