"""Explicit, bounded recovery with fresh task/session state after verified worker exit."""

import copy
import time
import uuid

from .evidence_inbox import EvidenceInbox
from .exposure_confirmation import ExposureClockMap
from .process_pair_capture import ProcessPairCapture
from .recent_evidence_handoff import RecentEvidenceHandoff
from .reserved_two_stage import ReservedTwoStageController
from .scheduled_pair_capture import ScheduledReaderFactory
from .two_stage_capture import TwoStageCaptureController
from .two_stage_runtime import TwoStageCaptureRuntime


class CaptureRecovery:
    """Single owner thread; start is explicit, never an automatic task replay.

    At most three generations. Old workers must be reaped before another factory
    is invoked. This only proves process isolation, not device recovery readiness.
    """

    def __init__(self, *, position_error_m, max_generations=3, capture_profile="immediate"):
        if type(max_generations) is not int or not 1 <= max_generations <= 3:
            raise ValueError("one to three generations required")
        if capture_profile not in ("immediate", "reserved"):
            raise ValueError("capture profile must be immediate or reserved")
        self.capture_profile = capture_profile
        self._handoff = None
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
        if self._handoff is not None:
            self._handoff.revoke()
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
                    "evidence_handoff": self._handoff.result() if self._handoff else None,
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
        self._handoff = None
        self._recorded = False
        self.state = "starting"
        try:
            self.backend = ProcessPairCapture(
                ScheduledReaderFactory(factory) if self.capture_profile == "reserved" else factory
            )
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
            cls = (
                ReservedTwoStageController
                if self.capture_profile == "reserved"
                else TwoStageCaptureController
            )
            controller = cls(
                0.02 if self.capture_profile == "reserved" else 0,
                ExposureClockMap(origin, 0),
                self.session,
                position_error_m=self.budget,
            )
            self.runtime = TwoStageCaptureRuntime(controller, capture_backend=self.backend)
            self._handoff = RecentEvidenceHandoff(controller, clock=self.runtime.clock)
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

    def consume_confirmation(self, session_id):
        """Deliver recent evidence once, before retirement; never authorize motion."""
        if self._handoff is None:
            return {
                "status": "refused",
                "reason": "no_active_handoff",
                "robot_action_authorized": False,
            }
        return self._handoff.consume(session_id)

    def queue_confirmation(self, session_id, inbox):
        """Consume directly into an inbox; a previously returned receipt cannot be requeued."""
        if not isinstance(inbox, EvidenceInbox):
            raise TypeError("EvidenceInbox required")
        if self._handoff is None:
            return {
                "status": "refused",
                "reason": "no_active_handoff",
                "robot_action_authorized": False,
            }
        return inbox.offer(self._handoff, session_id)

    def result(self):
        return copy.deepcopy(
            {
                "state": self.state,
                "capture_profile": self.capture_profile,
                "evidence_handoff": self._handoff.result() if self._handoff else None,
                "session_id": self.session,
                "attempts": self.attempts,
                "generation_limit": self.limit,
                "ignored_sessions": self.ignored_sessions,
                "history": self.history,
                "runtime": self.runtime.result() if self.runtime else None,
                "hardware_recovery_verified": False,
            }
        )
