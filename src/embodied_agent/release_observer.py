"""Opt-in shadow observation; never supplies commands or task success."""

from .release_confirmation import ReleaseConfirmation


class ReleaseObserver:
    def __init__(self):
        self.confirmation = None
        self.next_time = None
        self.rows = []

    def step(self, timestamp, events, locate):
        if self.confirmation is None:
            if not any(event["event"] == "intentional_release" for event in events):
                return
            self.confirmation = ReleaseConfirmation(timestamp)
            self.next_time = timestamp
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
        return self.confirmation.status if self.confirmation else "not_started"
