"""Lock-consistent, bounded audit exports; hashes are not signatures."""

import hashlib
import os
import tempfile
from pathlib import Path

from .budget_checkpoint import LIMIT, canonical, digest, fields, label, restore
from .checkpointed_runtime import checkpoint_lock
from .operation_inventory import MAX_ENTRIES, scan_receipt_keys
from .operation_receipt import validate_receipt
from .recovery_json import parse_recovery_json, read_recovery_json

MAX_RECEIPTS = 256
MAX_SOURCE_BYTES = 8 * LIMIT
MAX_SNAPSHOT = 16 * LIMIT


class SnapshotLimitExceeded(ValueError):
    pass


def _budget(envelope):
    fields(envelope, "state sha256")
    digest(envelope["sha256"])
    if hashlib.sha256(canonical(envelope["state"]).encode()).hexdigest() != envelope["sha256"]:
        raise ValueError("budget checksum mismatch")
    return restore(envelope["state"])


def _capture(path, validator, used):
    try:
        with path.open("rb") as stream:
            raw = stream.read(min(LIMIT, MAX_SOURCE_BYTES - used[0]) + 1)
        used[0] += len(raw)
        if used[0] > MAX_SOURCE_BYTES:
            raise SnapshotLimitExceeded("aggregate source budget exceeded")
        if len(raw) > LIMIT:
            return {"source_sha256": None, "data": None, "error": "SourceTooLarge"}
        source_hash = hashlib.sha256(raw).hexdigest()
        try:
            data = parse_recovery_json(raw.decode("utf-8"))
            validator(data)
        except (ValueError, TypeError) as exc:
            return {"source_sha256": source_hash, "data": None, "error": type(exc).__name__}
        return {"source_sha256": source_hash, "data": data, "error": None}
    except OSError as exc:
        return {"source_sha256": None, "data": None, "error": type(exc).__name__}


def _counts(items):
    counts = dict.fromkeys(("pending", "complete", "abandoned", "invalid"), 0)
    for item in items:
        state = "invalid" if item["error"] is not None else item["data"]["state"]
        counts[state] += 1
    return counts


def export_snapshot(checkpoint, output, *, source_id=None, capture_sequence=None):
    if source_id is not None:
        label(source_id)
    if capture_sequence is not None and (
        source_id is None or type(capture_sequence) is not int or capture_sequence < 1
    ):
        raise ValueError("capture sequence requires source ID and a positive integer")
    output = Path(output).absolute()
    with checkpoint_lock(checkpoint) as resolved:
        directory = resolved.with_name(resolved.name + ".operations")
        if output.resolve() in (resolved, resolved.with_name(resolved.name + ".lock")):
            raise ValueError("snapshot cannot replace source state or lock")
        if output.resolve().is_relative_to(directory):
            raise ValueError("snapshot output must be outside receipt directory")
        keys, ignored, scanned = scan_receipt_keys(directory)
        if len(keys) > MAX_RECEIPTS:
            raise SnapshotLimitExceeded("too many receipts for one snapshot")
        used = [0]
        budget = _capture(resolved, _budget, used)
        items = []
        for key in keys:
            item = _capture(directory / (key + ".json"), validate_receipt, used)
            item["receipt_key"] = key
            items.append(item)
        body = {
            "schema": 1,
            "budget": budget,
            "items": items,
            "counts": _counts(items),
            "scanned_entries": scanned,
            "ignored_entries": ignored,
            "source_bytes_read": used[0],
        }
        if source_id is not None:
            body.update(schema=2, source_id=source_id, capture_sequence=capture_sequence)
        envelope = {
            "snapshot": body,
            "sha256": hashlib.sha256(canonical(body).encode()).hexdigest(),
        }
        content = canonical(envelope)
        if len(content) > MAX_SNAPSHOT:
            raise SnapshotLimitExceeded("snapshot output exceeds limit")
        _verify_envelope(parse_recovery_json(content))
        # Publish a fully written file without overwriting an existing destination.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=output.parent, prefix=".snapshot-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {
        "sha256": envelope["sha256"],
        "source_id": body.get("source_id"),
        "capture_sequence": body.get("capture_sequence"),
        "counts": body["counts"],
        "has_errors": budget["error"] is not None or body["counts"]["invalid"] > 0,
        "ignored_entries": ignored,
    }


def read_snapshot(path):
    envelope = read_recovery_json(path, MAX_SNAPSHOT)
    _verify_envelope(envelope)
    return envelope


def verify_snapshot(path):
    return _verify_envelope(read_recovery_json(path, MAX_SNAPSHOT))


def _verify_envelope(envelope):
    fields(envelope, "snapshot sha256")
    digest(envelope["sha256"])
    body = envelope["snapshot"]
    if hashlib.sha256(canonical(body).encode()).hexdigest() != envelope["sha256"]:
        raise ValueError("snapshot checksum mismatch")
    if (
        not isinstance(body, dict)
        or type(body.get("schema")) is not int
        or body["schema"] not in (1, 2)
    ):
        raise ValueError("unsupported snapshot schema")
    fields(
        body,
        "schema budget items counts scanned_entries ignored_entries source_bytes_read"
        + (" source_id capture_sequence" if body["schema"] == 2 else ""),
    )
    if body["schema"] == 2:
        label(body["source_id"])
        sequence = body["capture_sequence"]
        if sequence is not None and (type(sequence) is not int or sequence < 1):
            raise ValueError("invalid snapshot sequence")
    if not isinstance(body["items"], list) or len(body["items"]) > MAX_RECEIPTS:
        raise ValueError("invalid snapshot items")
    for name in ("scanned_entries", "ignored_entries", "source_bytes_read"):
        if type(body[name]) is not int or body[name] < 0:
            raise ValueError("invalid snapshot counters")
    if (
        body["source_bytes_read"] > MAX_SOURCE_BYTES
        or body["scanned_entries"] != len(body["items"]) + body["ignored_entries"]
    ):
        raise ValueError("inconsistent snapshot counters")
    if body["scanned_entries"] > MAX_ENTRIES:
        raise ValueError("snapshot scan limit exceeded")
    fields(body["counts"], "pending complete abandoned invalid")
    if any(type(value) is not int or value < 0 for value in body["counts"].values()):
        raise ValueError("invalid snapshot state counts")
    keys = []
    for item, is_budget in [(body["budget"], True), *[(i, False) for i in body["items"]]]:
        fields(item, "source_sha256 data error" + ("" if is_budget else " receipt_key"))
        if not is_budget:
            digest(item["receipt_key"])
            keys.append(item["receipt_key"])
        if item["source_sha256"] is not None:
            digest(item["source_sha256"])
        if item["error"] is None:
            if item["source_sha256"] is None:
                raise ValueError("valid source requires digest")
            (_budget if is_budget else validate_receipt)(item["data"])
        elif not isinstance(item["error"], str) or not item["error"] or item["data"] is not None:
            raise ValueError("invalid snapshot error record")
    if keys != sorted(set(keys)) or body["counts"] != _counts(body["items"]):
        raise ValueError("snapshot keys or counts disagree")
    return {
        "sha256": envelope["sha256"],
        "source_id": body.get("source_id"),
        "capture_sequence": body.get("capture_sequence"),
        "counts": body["counts"],
        "has_errors": body["budget"]["error"] is not None or body["counts"]["invalid"] > 0,
        "ignored_entries": body["ignored_entries"],
    }
