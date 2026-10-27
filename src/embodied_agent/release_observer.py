"""Opt-in shadow observation; never supplies commands or task success."""

from .release_confirmation import ReleaseConfirmation
from .release_review import ReleaseReview


class ReleaseObserver:
    def __init__(self, *, allow_recheck=False):
        self.confirmation = None
        self.allow_recheck = allow_recheck
        self.review = None
        self.cancelled = False
        self.next_time = None
        self.rows = []

    def step(self, timestamp, events, locate, *, stop=False):
        if stop:
            if self.status in {"pending", "not_started"}:
                self.cancelled = True
                if self.review:
                    self.review.step(timestamp, locate, stop=True)
            return
        if self.cancelled:
            return
        if self.confirmation is None:
            if not any(event["event"] == "intentional_release" for event in events):
                return
            self.confirmation = ReleaseConfirmation(timestamp)
            if self.allow_recheck:
                self.review = ReleaseReview(timestamp)
                self.confirmation = self.review.initial
            self.next_time = timestamp
        if self.review:
            captured = []

            def observe():
                point, error = None, None
                try:
                    point = locate()
                except ValueError as exc:
                    error = str(exc)
                captured.append(
                    {
                        "time": timestamp,
                        "xyz": None if point is None else list(point),
                        "error": error,
                    }
                )
                return point

            self.review.step(timestamp, observe)
            if captured:
                self.rows.append(
                    {**captured[0], "status": self.review.status, "reason": self.review.reason}
                )
            return
        if self.confirmation.status != "pending" or timestamp + 1e-9 < self.next_time:
            return
        point, error = None, None
        try:
            point = locate()
        except ValueError as exc:
            error = str(exc)
        status = self.confirmation.observe(timestamp, point)
        self.rows.append(
            {
                "time": timestamp,
                "xyz": None if point is None else list(point),
                "error": error,
                "status": status,
                "reason": self.confirmation.reason,
            }
        )
        self.next_time = timestamp + 0.2

    @property
    def status(self):
        if self.cancelled:
            return "cancelled"
        if self.review:
            return self.review.status
        return self.confirmation.status if self.confirmation else "not_started"
