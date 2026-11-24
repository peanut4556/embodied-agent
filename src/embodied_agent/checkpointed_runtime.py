"""Single-host, cooperative locked execution with a prewritten quarantine snapshot."""

import copy
from contextlib import contextmanager
from pathlib import Path

from .action_budget import seconds
from .budget_checkpoint import load_budget, save_budget

MOTION_ACTIONS = frozenset({"locate", "pick", "place", "verify"})


class CheckpointBusy(RuntimeError):
    pass


@contextmanager
def checkpoint_lock(path):
    # POSIX only; importing here leaves ordinary runtime usable on other platforms.
    import fcntl

    path = Path(path).resolve()
    with path.with_name(path.name + ".lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CheckpointBusy("budget checkpoint is in use") from exc
        try:
            yield path
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    # Never unlink the sidecar: replacing its inode could split the lock domain.


def run_checkpointed(runtime, instruction, path, *, task_timeout_s, stop_requested=None):
    seconds(task_timeout_s)
    if stop_requested is not None and not callable(stop_requested):
        raise ValueError("stop_requested must be callable")
    with checkpoint_lock(path) as resolved:
        budget = load_budget(resolved)
        if budget.drift_policy != "quarantine":
            raise ValueError("checkpointed execution requires quarantine policy")
        # Persist uncertainty before any runtime call. Abrupt death leaves this snapshot.
        uncertain = copy.deepcopy(budget)
        uncertain.degraded.update(MOTION_ACTIONS)
        uncertain.degraded.update(budget.estimates)
        save_budget(uncertain, resolved)
        result = runtime.run(
            instruction,
            task_timeout_s=task_timeout_s,
            stop_requested=stop_requested,
            action_budget=budget,
        )
        # On exceptions, the conservative snapshot remains; never auto-clear it.
        save_budget(budget, resolved)
        return result
