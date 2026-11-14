"""Single-use, expiring evidence delivery; never a robot action permission."""

import copy
import math
import time
import uuid

from .two_stage_capture import TwoStageCaptureController


class RecentEvidenceHandoff:
    """One owner thread, one handoff per in-process controller instance.

    A successful historical result can remain in the audit forever, but delivery
    as recent evidence requires a fresh clock check AFTER snapshot construction.
    Expiry cannot be renewed by polling, copying, or recreating this handoff.
    This is an application lifecycle contract, not an anti-tampering boundary.
    """

    def __init__(self, controller, *, clock=time.monotonic):
        if not isinstance(controller, TwoStageCaptureController) or not callable(clock):
            raise TypeError("two-stage controller and callable host clock required")
        if getattr(controller, "_recent_evidence_claim", False):
            raise RuntimeError("controller evidence already assigned to a handoff")
        controller._recent_evidence_claim = True
        self.controller, self.clock = controller, clock
        self.session = controller.candidate.check.session
        self.mapping = controller.mapping
        self.state = "available"
        self.revoked = False
        self.attempts = 0
        self.receipt = None

    def revoke(self):
        self.revoked = True
        if self.state == "available":
            self.state = "revoked"

    def consume(self, session_id):
        self.attempts += 1

        def refused(reason):
            return {"status": "refused", "reason": reason, "robot_action_authorized": False}

        if session_id != self.session:
            return refused("session_mismatch")
        if self.state != "available":
            return refused("already_consumed" if self.state == "consumed" else self.state)
        c = self.controller
        if c.status != "confirmed_recent_evidence":
            return refused("confirmation_not_complete")
        # Snapshot copying is deliberately inside the age budget.
        receipt_id = uuid.uuid4().hex
        evidence = c.result()
        last_exposure = (
            self.mapping.host_origin
            + (evidence["witness"]["last_exposure_s"] - self.mapping.task_origin)
            / self.mapping.rate
        )
        expires = last_exposure + 0.02 / self.mapping.rate
        try:
            host = self.clock()
            if (
                isinstance(host, bool)
                or not isinstance(host, (int, float))
                or not math.isfinite(host)
                or host < max(self.mapping.host_origin, c.last_host, last_exposure)
            ):
                raise ValueError("invalid or regressed host clock")
        except Exception:  # noqa: BLE001 - fail closed at the injected clock boundary
            self.state = "invalid_clock"
            return refused(self.state)
        if host >= expires:
            self.state = "expired"
            return refused(self.state)
        self.state = "consumed"
        self.receipt = {
            "receipt_id": receipt_id,
            "status": "consumed",
            "reason": "fresh_evidence_consumed",
            "session_id": self.session,
            "consumed_host_time": host,
            "last_exposure_host_time": last_exposure,
            "expires_host_time": expires,
            "remaining_s": expires - host,
            "robot_action_authorized": False,
        }
        return dict(self.receipt, evidence=evidence)

    def result(self):
        return copy.deepcopy(
            {
                "state": self.state,
                "revoked": self.revoked,
                "attempts": self.attempts,
                "receipt": self.receipt,
            }
        )
