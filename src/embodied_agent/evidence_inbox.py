"""One-slot in-process evidence queue with checks at consumer dispatch time."""

import copy
import math
import time
import uuid
from collections import deque

from .recent_evidence_handoff import RecentEvidenceHandoff


class EvidenceInbox:
    """One owner thread. Queued evidence is never exposed through queue snapshots.

    All clocks must share the same host monotonic domain. This prevents stale
    dispatch through this API, not copied-data replay or physical motion hazards.
    A consumer is invoked at most once per queued receipt; exceptions leave an
    unknown consumer outcome and must not trigger an automatic replay.
    """

    def __init__(self, *, clock=time.monotonic):
        if not callable(clock):
            raise TypeError("callable host clock required")
        self.clock = clock
        self._pending = None
        self._busy = False
        self._closed = False
        self._clock_failed = False
        self._last_host = None
        self._history = deque(maxlen=16)
        self._completed = 0

    def _event(self, status, reason, **fields):
        return {"status": status, "reason": reason, "robot_action_authorized": False, **fields}

    def offer(self, handoff, session_id):
        if self._closed or self._clock_failed:
            return self._event("refused", "inbox_closed" if self._closed else "invalid_clock")
        if self._busy or self._pending is not None:
            return self._event("refused", "inbox_busy")
        if not isinstance(handoff, RecentEvidenceHandoff):
            raise TypeError(
                "live RecentEvidenceHandoff required; receipt dictionaries are not accepted"
            )
        receipt = handoff.consume(session_id)
        if receipt["status"] != "consumed":
            return receipt
        ticket = uuid.uuid4().hex
        self._pending = (ticket, handoff, receipt)
        return self._event("queued", "awaiting_consumer", ticket=ticket)

    def _finish(self, event):
        self._history.append(dict(event))
        self._completed += 1
        self._pending = None
        return event

    def dispatch(self, consumer):
        if self._busy:
            return self._event("refused", "inbox_busy")
        if self._pending is None:
            return self._event("refused", "no_pending_evidence")
        if not callable(consumer):
            raise TypeError("callable evidence consumer required")
        ticket, issuer, receipt = self._pending
        meta = {"ticket": ticket, "session_id": receipt["session_id"]}
        if issuer.revoked:
            return self._finish(self._event("discarded", "session_revoked", **meta))
        try:
            host = self.clock()
            if (
                isinstance(host, bool)
                or not isinstance(host, (int, float))
                or not math.isfinite(host)
                or host < receipt["consumed_host_time"]
                or (self._last_host is not None and host < self._last_host)
            ):
                raise ValueError("invalid consumer clock")
        except Exception:  # noqa: BLE001 - clock boundary fails closed
            self._clock_failed = True
            return self._finish(self._event("discarded", "invalid_clock", **meta))
        self._last_host = host
        if host >= receipt["expires_host_time"]:
            return self._finish(self._event("discarded", "expired_in_queue", **meta))
        # Burn the slot before user code. Neither recursion nor a raised exception
        # can re-invoke the consumer for this receipt.
        self._pending = None
        self._busy = True
        try:
            consumer(receipt)
        except Exception as exc:  # noqa: BLE001 - consumer may already have had effects
            event = self._event(
                "outcome_unknown",
                "consumer_raised",
                error=f"{type(exc).__name__}: {exc}"[:512],
                dispatch_host_time=host,
                **meta,
            )
        else:
            event = self._event("delivered", "consumer_returned", dispatch_host_time=host, **meta)
        finally:
            self._busy = False
        return self._finish(event)

    def close(self):
        self._closed = True
        if self._pending is not None:
            ticket, _, receipt = self._pending
            self._finish(
                self._event(
                    "discarded", "inbox_closed", ticket=ticket, session_id=receipt["session_id"]
                )
            )

    def result(self):
        pending = None
        if self._pending is not None:
            ticket, _, receipt = self._pending
            pending = {
                "ticket": ticket,
                "session_id": receipt["session_id"],
                "expires_host_time": receipt["expires_host_time"],
            }
        return copy.deepcopy(
            {
                "closed": self._closed,
                "clock_failed": self._clock_failed,
                "busy": self._busy,
                "pending": pending,
                "completed": self._completed,
                "history": list(self._history),
                "history_limit": 16,
            }
        )
