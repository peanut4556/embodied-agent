"""Single-host, cooperative locked execution with a prewritten quarantine snapshot."""

import copy
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
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


def _execute(
    runtime, instruction, path, *, task_timeout_s, stop_requested, state, already_locked=False
):
    state.update(phase="validate", execution_started=False, checkpoint_state="unchanged")
    seconds(task_timeout_s)
    if stop_requested is not None and not callable(stop_requested):
        raise ValueError("stop_requested must be callable")
    state["phase"] = "lock"
    with nullcontext(Path(path).resolve()) if already_locked else checkpoint_lock(path) as resolved:
        state["phase"] = "load"
        budget = load_budget(resolved)
        state["phase"] = "policy"
        if budget.drift_policy != "quarantine":
            raise ValueError("checkpointed execution requires quarantine policy")
        uncertain = copy.deepcopy(budget)
        uncertain.degraded.update(MOTION_ACTIONS)
        uncertain.degraded.update(budget.estimates)
        state.update(phase="prepare", checkpoint_state="not_confirmed")
        save_budget(uncertain, resolved)
        state.update(phase="execute", execution_started=True, checkpoint_state="prepared")
        result = runtime.run(
            instruction,
            task_timeout_s=task_timeout_s,
            stop_requested=stop_requested,
            action_budget=budget,
        )
        state.update(phase="commit", checkpoint_state="not_confirmed")
        save_budget(budget, resolved)
        state.update(phase="unlock", checkpoint_state="committed")
    state["phase"] = "complete"
    return result


def run_checkpointed(runtime, instruction, path, *, task_timeout_s, stop_requested=None):
    """Compatibility API: propagate original exceptions."""
    return _execute(
        runtime,
        instruction,
        path,
        task_timeout_s=task_timeout_s,
        stop_requested=stop_requested,
        state={},
    )


def run_checkpointed_report(runtime, instruction, path, *, task_timeout_s, stop_requested=None):
    """Structured boundary for controllers; never recommends automatic replay."""
    state = {}
    try:
        result = _execute(
            runtime,
            instruction,
            path,
            task_timeout_s=task_timeout_s,
            stop_requested=stop_requested,
            state=state,
        )
    except Exception as exc:  # noqa: BLE001 - explicit controller error boundary
        return {
            "success": False,
            "task_result": None,
            "checkpoint": dict(state),
            "error": {
                "code": "checkpoint_busy"
                if isinstance(exc, CheckpointBusy)
                else "checkpoint_" + state["phase"] + "_failed",
                "cause_type": type(exc).__name__,
                "action_outcome_unknown": state["execution_started"],
            },
        }
    return {
        "success": result.success,
        "task_result": asdict(result),
        "checkpoint": dict(state),
        "error": None,
    }
