"""Cooperative duplicate-request suppression, not physical exactly-once execution."""

import hashlib
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .action_budget import seconds
from .budget_checkpoint import LIMIT, canonical, fields, label
from .checkpointed_runtime import CheckpointBusy, _execute, checkpoint_lock
from .recovery_json import read_recovery_json


def _write(path, value):
    value = dict(value)
    value["sha256"] = hashlib.sha256(canonical(value).encode()).hexdigest()
    content = canonical(value)
    if len(content) > LIMIT:
        raise ValueError("operation receipt exceeds limit")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=".receipt-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_operation(runtime, instruction, path, *, operation_id, task_timeout_s, request_context):
    state = {
        "phase": "operation_validate",
        "execution_started": False,
        "checkpoint_state": "unchanged",
    }
    code = None
    prior_execution_possible = False
    try:
        label(operation_id)
        seconds(task_timeout_s)
        if not isinstance(instruction, str) or not isinstance(request_context, dict):
            raise TypeError("instruction and request_context required")
        request = canonical(
            {
                "instruction": instruction,
                "timeout_s": float(task_timeout_s),
                "context": request_context,
            }
        )
        fingerprint = hashlib.sha256(request.encode()).hexdigest()
        state["phase"] = "operation_lock"
        with checkpoint_lock(path) as resolved:
            directory = resolved.with_name(resolved.name + ".operations")
            directory.mkdir(exist_ok=True)
            receipt = directory / (hashlib.sha256(operation_id.encode()).hexdigest() + ".json")
            state["phase"] = "operation_lookup"
            if receipt.exists():
                prior_execution_possible = True
                state["checkpoint_state"] = "not_confirmed"
                stored = read_recovery_json(receipt, LIMIT)
                fields(stored, "schema request_sha256 state result sha256")
                checksum = stored.pop("sha256")
                if hashlib.sha256(canonical(stored).encode()).hexdigest() != checksum:
                    raise ValueError("receipt checksum mismatch")
                if type(stored["schema"]) is not int or stored["schema"] != 1:
                    raise ValueError("unsupported receipt schema")
                if stored["request_sha256"] != fingerprint:
                    code = "operation_conflict"
                    raise ValueError("operation ID already bound to another request")
                if stored["state"] == "pending" and stored["result"] is None:
                    prior_execution_possible = True
                    state["checkpoint_state"] = "not_confirmed"
                    code = "operation_in_doubt"
                    raise ValueError("operation has no committed result")
                if stored["state"] != "complete":
                    raise ValueError("invalid receipt state")
                result = stored["result"]
                fields(result, "success message completed_steps confirmation_reports")
                if (
                    type(result["success"]) is not bool
                    or type(result["completed_steps"]) is not int
                    or result["completed_steps"] < 0
                    or not isinstance(result["message"], str)
                    or not isinstance(result["confirmation_reports"], list)
                ):
                    raise ValueError("invalid cached task result")
                return {
                    "success": result["success"],
                    "task_result": result,
                    "checkpoint": {
                        "phase": "historical",
                        "execution_started": False,
                        "checkpoint_state": "historical_committed",
                    },
                    "error": None,
                    "operation": {"id": operation_id, "replayed": True},
                }
            state["phase"] = "operation_register"
            stored = {
                "schema": 1,
                "request_sha256": fingerprint,
                "state": "pending",
                "result": None,
            }
            _write(receipt, stored)
            result = _execute(
                runtime,
                instruction,
                resolved,
                task_timeout_s=task_timeout_s,
                stop_requested=None,
                state=state,
                already_locked=True,
            )
            state["phase"] = "operation_commit"
            stored.update(state="complete", result=asdict(result))
            _write(receipt, stored)
        state["phase"] = "complete"
        return {
            "success": result.success,
            "task_result": asdict(result),
            "checkpoint": dict(state),
            "error": None,
            "operation": {"id": operation_id, "replayed": False},
        }
    except Exception as exc:  # noqa: BLE001 - controller boundary
        return {
            "success": False,
            "task_result": None,
            "checkpoint": dict(state),
            "error": {
                "code": code
                or (
                    "checkpoint_busy"
                    if isinstance(exc, CheckpointBusy)
                    else state["phase"] + "_failed"
                ),
                "cause_type": type(exc).__name__,
                "action_outcome_unknown": state["execution_started"] or prior_execution_possible,
            },
            "operation": {"id": operation_id, "replayed": False},
        }
