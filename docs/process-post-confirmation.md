# 确认后监测的采集进程接入与回收

2026-11-02。新增实验性 ProcessPostConfirmation，将独立区域监测与 CameraProcessSession 接通。支持从成功的 ProcessExposureObserver / RecoveryExposureObserver 显式交接已有会话，继续采集后续新帧，保留原始释放确认快照。

## 交接与轮询

`ProcessPostConfirmation.from_confirmed(observer)` 要求原控制器确已 confirmed_in_tray、具有确认器及任务时间，且会话标识仍匹配。相机会话须为 idle 或 ready，不能带着旧的待完成请求交接。交接起点是原确认的接收任务时间，不会重新确认机械任务。

原成功帧已经被原控制器消费，适配层不把 ready 状态当作一帧新数据；它发出新请求，仅当待完成请求产生 ready/error 结果后才更新监测。之后保持至多一个在途请求。

频繁 poll 时，没有完成结果只是等待，不打断连续区域外证据。但如果上一次有效曝光已超过映射的 max_age（默认 250 ms），等待期间也会将观测状态降为 unknown。明确返回 None 或错误同样打断连续性。已有区域外历史告警不会被清除。

停止、任务到期和会话变更先于 IPC 轮询及新请求处理。采集超时、异常退出、已关闭或清理失败会使本次监测 invalidated；新请求发送失败也会立即使监测失效。该入口不自动重启相机，不跨会话沿用映射或历史证据。

## 资源与调用方式

```python
# observer 已完成视觉确认，且其 session 尚未关闭。
with ProcessPostConfirmation.from_confirmed(observer) as followup:
    while followup.status == "monitoring":
        followup.step(current_task_time(), stop=stop_requested())
        # 由调用方以适合采集的节奏继续调度。
result = followup.result()
```

交接后调用方必须让 followup 独占采集会话，不再由旧 observer 调用采集、重启或 close；Python 对象本身不实施跨线程所有权锁。

step 不执行进程 join。停止、到期或故障后需要显式 close，或退出 with 上下文，才实际回收子进程。close/上下文退出属于生命周期管理操作，包含既有 CameraProcessSession 的有界清理等待，不应放入高频伺服循环。

result 保留独立监测结果，并增加 cleanup（not_requested / succeeded / failed）与 cleanup_required。清理异常保存错误后重新抛出，不伪装成成功关闭；调用方可以重试 close。即便监测已 completed，也必须单独确认清理成功。

该模块仍是显式实验入口，生产默认路径未启用。它不能证明物体掉出托盘，也不执行补救抓取或机械动作。调用方若完全停止调度，仍只能取得上一次调用时的快照。

## 验证

复现：`make process-post-confirmation-test`。

新增 10 项测试：

- 等待轮询不重复消费帧或重置区域外连续性。
- 在途请求长时间无新数据时，旧的区域内证据过期转为 unknown。
- 明确缺失帧打断连续性。
- 停止、到期、会话变更不再 poll/request，并要求显式清理。
- 清理失败记录与成功重试。
- 请求失败立即失效。
- 拒绝未确认或带在途请求的交接。
- 实际 spawn 相机桩完成一秒稳定确认后交接，取得晚于确认时刻的新曝光，停止并在上下文退出时回收；原确认结果不变。
- 实际阻塞和退出码 7 的崩溃分别使监测失效，随后回收进程。
- 实际进程在监测窗口到期后停止请求，上下文退出回收，旧位置不再显示有效。

共 223 项相关回归通过（58 核心、31 接触、34 释放、13 线程采集、14 曝光、7 采集进程、8 进程恢复、39 恢复相关、9 独立后续监测、10 进程后续监测），Ruff 与差异检查通过。

实际进程测试中的设备仍是相机桩；没有连接真实相机，没有新的物理抓放、训练或统计延迟上界结论。
