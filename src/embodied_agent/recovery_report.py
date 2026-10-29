"""Serializable diagnostic snapshots; observation recovery is not task success."""

import copy
import json
from pathlib import Path

_REASONS = {
    "restart_limit_reached": "重启次数已用尽",
    "insufficient_recovery_budget": "剩余任务时间不足以尝试恢复",
    "delivery_deadline_reached": "原始确认期限已到",
    "camera_cleanup_failed": "旧采集进程未能回收",
    "camera_restart_failed": "采集进程重启失败",
    "invalid_recovery_clock": "恢复期间任务时钟无效或回退",
    "camera_unavailable": "采集进程不可用，等待恢复决策",
    "awaiting_clock_mapping": "等待当前会话的新时钟映射",
    "collecting_evidence": "正在累计新的稳定观测证据",
    "stable_visual_evidence": "新的视觉稳定证据满足确认条件",
    "user_stop": "用户已停止确认",
    "observer_closed": "观测控制器已关闭",
    "observed_outside_confirmation_region": "观测位置在保守确认区域外",
}


def recovery_report(observer):
    """Snapshot on the observer's owner thread; never poll, restart or close it."""
    result = observer.result()
    attempts = result["recovery_attempts"]
    restored = [a for a in attempts if a["outcome"] == "observation_restored"]
    report = {
        "schema_version": 1,
        "scope": "experimental_visual_release_confirmation",
        "result": result,
        "diagnosis": {
            "observation_restored": bool(restored),
            "restored_attempts": [a["attempt"] for a in restored],
            "visual_confirmation_succeeded": result["status"] == "confirmed_in_tray",
            "terminal": result["status"] != "pending",
            "reason_code": result["reason"],
            "explanation": _REASONS.get(result["reason"], "请结合原因代码与逐帧记录诊断"),
            "cleanup_required": observer.cleanup["status"] != "succeeded",
        },
        "cleanup": observer.cleanup,
        "camera": {
            "session": observer.session.session,
            "status": observer.session.status,
            "error": observer.session.error,
        },
        "policy": {"min_remaining": observer.min_remaining},
        "events": observer.rows,
    }
    # Detach nested event/cleanup structures as well as attempt records.
    return copy.deepcopy(report)


def write_recovery_report(path, report):
    """Create a UTF-8 JSON report without replacing existing experiment evidence.

    Serialize first: invalid/non-finite data must not leave a new output file.
    Parent directories must already exist; filesystem errors propagate.
    """
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload)
