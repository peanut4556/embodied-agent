"""Bounded, unambiguous JSON input for recovery metadata."""

import json
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate recovery JSON field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonstandard recovery JSON constant")


def read_recovery_json(path, limit):
    with Path(path).open(encoding="utf-8") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("recovery JSON exceeds size limit")
    return parse_recovery_json(raw)


def parse_recovery_json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        pending = [(value, 0)]
        while pending:
            node, depth = pending.pop()
            if depth > 64:
                raise ValueError("recovery JSON nesting exceeds 64 levels")
            if isinstance(node, dict):
                pending.extend((child, depth + 1) for child in node.values())
            elif isinstance(node, list):
                pending.extend((child, depth + 1) for child in node)
        return value
    except RecursionError as exc:
        raise ValueError("recovery JSON nesting exceeds parser limit") from exc
