"""Bounded read-only recovery audit over one source snapshot."""

import argparse
import json
from collections import Counter

from embodied_agent.checkpoint_store import _load, _read_store, _separate
from embodied_agent.evidence_checkpoint import _read, _verify
from embodied_agent.indexed_evidence_ledger import _discover_connection


def audit_evidence(database, store, *, max_records=10000):
    if type(max_records) is not int or not 1 <= max_records <= 100000:
        raise ValueError("max_records must be 1..100000")
    _separate(store, database)

    def audit_store(connection):
        head = _load(connection)
        if head["status"] != "loaded":
            return head

        def audit_source(source):
            failure = _verify(source, head["checkpoint"])
            if failure is not None:
                return failure
            counts = Counter()
            examined, cursor = 0, 0
            while examined < max_records:
                page = _discover_connection(source, cursor, min(100, max_records - examined))
                if not page["entries"]:
                    break
                for entry in page["entries"]:
                    counts[entry["reconciliation"]["status"]] += 1
                examined += len(page["entries"])
                cursor = page["next_after"]
            more = bool(
                source.execute(
                    "SELECT 1 FROM evidence_intents_v1 WHERE sequence>? LIMIT 1", (cursor,)
                ).fetchone()
            )
            # Reverse traversal finds records absent from the index, including
            # additions beyond the checkpoint. Keep both scans in this snapshot.
            reverse_counts = Counter()
            records_examined, record_cursor = 0, None
            while records_examined < max_records:
                rows = source.execute(
                    "SELECT r.rowid, r.session_id, r.payload_sha256, i.receipt_id, "
                    "i.session_id, i.payload_sha256 FROM recent_evidence_v1 r "
                    "LEFT JOIN evidence_intents_v1 i ON i.receipt_id=r.receipt_id "
                    "WHERE (? IS NULL OR r.rowid>?) ORDER BY r.rowid LIMIT ?",
                    (record_cursor, record_cursor, min(100, max_records - records_examined)),
                ).fetchall()
                if not rows:
                    break
                for rowid, session, digest, identity, indexed_session, indexed_digest in rows:
                    if identity is None:
                        reverse_counts["unindexed"] += 1
                    elif (session, digest) != (indexed_session, indexed_digest):
                        reverse_counts["binding_conflict"] += 1
                    else:
                        reverse_counts["indexed"] += 1
                    record_cursor = rowid
                records_examined += len(rows)
            records_more = bool(
                source.execute(
                    "SELECT 1 FROM recent_evidence_v1 WHERE (? IS NULL OR rowid>?) LIMIT 1",
                    (record_cursor, record_cursor),
                ).fetchone()
            )
            if (
                reverse_counts["unindexed"]
                or reverse_counts["binding_conflict"]
                or any(status not in ("recorded", "not_found") for status in counts)
            ):
                status = "blocked"
            elif more or records_more:
                status = "incomplete"
            elif counts["not_found"]:
                status = "pending"
            else:
                status = "verified"
            return {
                "status": status,
                "examined": examined,
                "counts": dict(counts),
                "scan_complete": not more and not records_more,
                "index_scan_complete": not more,
                "record_scan_complete": not records_more,
                "records_examined": records_examined,
                "record_counts": dict(reverse_counts),
                "checkpoint_generation": head["generation"],
                "checkpoint_history_verified": True,
                "checkpoint_record_count": head["checkpoint"]["records"]["count"],
                "robot_action_authorized": False,
            }

        return _read(database, audit_source)

    result = _read_store(store, audit_store)
    if result["status"] not in ("verified", "pending", "incomplete", "blocked"):
        result = {"status": "blocked", "cause": result, "robot_action_authorized": False}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--max-records", type=int, default=10000)
    args = parser.parse_args(argv)
    try:
        result = audit_evidence(args.database, args.store, max_records=args.max_records)
    except (ValueError, OSError) as exc:
        result = {"status": "blocked", "error": str(exc)[:512], "robot_action_authorized": False}
    result["guidance"] = {
        "verified": "历史基线匹配，双向扫描完成且索引与证据对应；不表示机器人动作成功或新增记录已受检查点保护。",
        "pending": "存在未落账索引；保留原始状态核查，不自动重放。",
        "incomplete": "扫描达到数量上限，不能判断全库；可提高上限重新执行。",
        "blocked": "恢复核验失败；保留数据库和检查点，不自动重新登记。",
    }[result["status"]]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return {"verified": 0, "pending": 3, "incomplete": 4, "blocked": 5}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
