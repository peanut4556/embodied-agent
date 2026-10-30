# 恢复报告的曝光证据离线重放

2026-10-30。在报告字段自洽审计之外，新增曝光证据重放：用新的 ExposureConfirmation 实例读取报告内的观测输入，重新计算逐帧判断，核对最终视觉成功声明。生产默认路径未变化，不访问相机或发送机械动作。

## 新增记录

新恢复报告保留顶层 schema_version 1，增加 evidence 元数据：证据格式版本、算法标识 exposure_confirmation_v1 和原始释放任务起点。事件中增加：

- mapping：绑定时的任务时间、主机时间、会话启动时间及全部仿射映射参数。
- observation：原始三维坐标（或 None）、主机域曝光时间、主机接收时间，以及既有逐帧结果。接收时间直接复用实际传给确认器的值，避免记录时再次读时钟造成偏差。
- deadline：原始任务期限到达的时间。沿用已有会话失效、重启等待、停止与关闭事件。

记录中保留缺失或被拒绝的观测，使重放能再现证据连续性中断。时间、位置数据均来自运行中的控制器，不在导出时补造。

## 复核规则与兼容性

重放检查映射绑定、会话匹配、任务时间递增、原始期限、缺失帧以及重置边界；同一确认器实例不得跨失效事件继续累计。每帧重新计算的状态、原因、错误与有效曝光时间必须与记录一致，最终确认成功声明必须由重放得到。恢复尝试标为 observation_restored 时，还须在对应新会话找到被接受的曝光证据。

未知证据版本、未知算法标识、缺失必要字段及终态后的新事件均拒绝。该逻辑已接入既有离线审计：新报告只要重放失败，该场景就不会进入有效汇总。

没有 evidence 字段的历史报告继续接受原有字段自洽检查，但 evidence_replay.status 明确为 unavailable，并给出 legacy_report_without_raw_evidence 原因。具有可处理证据的报告返回 verified、重放帧数和是否复现视觉成功；零帧报告可通过结构及“未宣称视觉成功”的检查，其重放帧数仍为 0。汇总新增 replay_counts，分别统计 verified / unavailable。

这里使用同一确认算法的新实例，并非独立实现的数学判据。它验证记录可复算，不证明算法正确、报告真实或数据来自实际设备。删除 evidence 字段会使报告降级为 unavailable，不应将这种报告作为通过证据复核的样本。当前记录也不能复核驱动原始图像、IPC 请求边界或清理动作的物理效果。

## 结果与复现

- [新证据报告](evaluations/recovery-evidence-v1.json)：8 个确定性场景，包括恢复但未确认、确认成功、预算拒绝、次数耗尽、失败、停止和清理异常。
- [新旧兼容性审计](evaluations/recovery-replay-v1.json)：8 个新场景 verified、8 个旧场景 unavailable、拒绝 0。两组是同类场景的不同记录格式，不是 16 次独立可靠性实验；汇总总数仅用于兼容性检查。

```sh
make recovery-replay-test
make recovery-replay-check
# 新运行的场景记录，可自行指定未占用输出路径：
PYTHONPATH=src:. .venv/bin/python scripts/check_recovery_reports.py --output outputs/new-evidence.json
PYTHONPATH=src:. .venv/bin/python scripts/audit_recovery_reports.py outputs/new-evidence.json --output outputs/new-evidence-audit.json
```

新增 8 项测试覆盖完整 JSON 往返、坐标改动、关键帧删除、时间/映射/会话篡改、汇总自洽但证据不足的虚假成功、重置与终态边界、旧报告降级、未知版本及截断记录。相关回归共 192 项通过（58 核心、31 接触、22 释放、13 线程采集、14 曝光、7 采集进程、8 进程恢复、39 恢复相关），Ruff 和差异检查通过。未连接真实设备，未新增物理抓放或训练。
