"""Experimental post-release RGB-D evidence, separate from task success."""

import numpy as np


class ReleaseConfirmation:
    """Consume fresh timestamped positions after an externally observed release."""

    def __init__(self, start_time):
        if not np.isfinite(start_time):
            raise ValueError("finite start time required")
        self.start = float(start_time)
        self.last_time = None
        self.previous = None
        self.since = None
        self.status = "pending"

    def observe(self, timestamp, xyz):
        if not np.isfinite(timestamp) or timestamp < self.start:
            raise ValueError("invalid observation time")
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("observations must be fresh and ordered")
        point = None if xyz is None else np.asarray(xyz, dtype=float)
        if point is not None and (point.shape != (3,) or not np.isfinite(point).all()):
            raise ValueError("invalid observed position")
        if self.status != "pending":
            return self.status
        gap = None if self.last_time is None else timestamp - self.last_time
        inside = point is not None and (
            abs(point[0] - 0.64) < 0.05 and abs(point[1]) < 0.04 and 0.032 < point[2] < 0.058
        )
        stable = (
            inside
            and self.previous is not None
            and gap is not None
            and gap <= 0.25
            and np.linalg.norm(point - self.previous) / gap < 0.02
        )
        if not stable:
            self.since = timestamp if inside else None
        elif self.since is None:
            self.since = self.last_time
        # No success from observations beyond the bounded confirmation window.
        if (
            timestamp - self.start <= 3 + 1e-9
            and self.since is not None
            and timestamp - self.since >= 1 - 1e-9
        ):
            self.status = "confirmed_in_tray"
        elif timestamp - self.start >= 3 - 1e-9:
            self.status = "outside_tray" if point is not None and not inside else "unconfirmed"
        self.last_time, self.previous = timestamp, point
        return self.status
