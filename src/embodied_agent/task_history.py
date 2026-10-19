"""Bounded local task history with validated loads and atomic replacement."""

import json
import math
import os
import tempfile
from pathlib import Path


class TaskHistoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.warning = ""
        self.blocked = False
        self.saved = None

    @staticmethod
    def encode(tasks):
        if not isinstance(tasks, list) or len(tasks) > 20:
            raise ValueError("history must contain at most 20 tasks")
        ids = set()
        for task in tasks:
            if not isinstance(task, dict):
                raise TypeError("invalid task record")
            run_id = task.get("run_id")
            if not isinstance(run_id, str) or not run_id or run_id in ids:
                raise ValueError("invalid or duplicate run id")
            ids.add(run_id)
            if task.get("phase") not in {"success", "failed", "stopped"}:
                raise ValueError("only finished tasks may be stored")
            for key in ("started_at", "finished_at"):
                value = task.get(key)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("invalid task timestamp")
            if not isinstance(task.get("plan"), list) or not isinstance(task.get("events"), list):
                raise TypeError("invalid plan or events")
            for event in task["events"]:
                if not isinstance(event, dict) or any(
                    not isinstance(event.get(k), str) for k in ("time", "text")
                ):
                    raise ValueError("invalid event")
            for key in ("message", "instruction", "stop_error"):
                if key in task and not isinstance(task[key], str):
                    raise ValueError("invalid task text")
            failure = task.get("failure")
            if failure is not None and (
                not isinstance(failure, dict)
                or any(not isinstance(failure.get(k), str) for k in ("title", "detail", "recovery"))
            ):
                raise ValueError("invalid failure detail")
        payload = json.dumps(
            {"version": 1, "tasks": tasks}, ensure_ascii=False, allow_nan=False, sort_keys=True
        )
        if len(payload.encode()) > 2_000_000:
            raise ValueError("history exceeds 2 MB")
        return payload

    def load(self):
        try:
            if self.path.stat().st_size > 2_000_000:
                raise ValueError("history exceeds 2 MB")
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("unsupported history version")
            self.saved = self.encode(data.get("tasks"))
            return data["tasks"]
        except FileNotFoundError:
            return []
        except (OSError, ValueError, TypeError) as exc:
            # Preserve unrecognized/corrupt data instead of silently overwriting it.
            self.blocked = True
            self.warning = (
                f"历史文件无法读取，本次新增记录仅保存在内存；请备份并修复文件后重启控制台：{exc}"
            )
            return []

    def save(self, tasks):
        if self.blocked:
            return
        temporary = None
        try:
            payload = self.encode(tasks)
            if payload == self.saved:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=".task-history-",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            self.saved, self.warning = payload, ""
        except (OSError, ValueError, TypeError) as exc:
            self.warning = f"任务记录尚未保存到磁盘，当前可查看或导出内存记录：{exc}"
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
