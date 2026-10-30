# 相机恢复诊断报告

2026-10-29。新增独立诊断快照和 JSON 导出接口，将观测恢复、视觉确认与进程清理分别呈现。保持原有实验性恢复策略与生产默认路径不变。

## 报告含义

`recovery_report(observer)` 必须在控制器的拥有者线程调用；只读取状态，不轮询相机、不重启、不关闭进程。它复制逐帧事件、恢复尝试和清理状态，后续控制器变化或调用方修改报告均不影响另一方。

报告 schema_version 为 1；内部 result 保留恢复控制器自身的 schema_version 2。主要字段：

| 字段 | 含义 |
| --- | --- |
| diagnosis.observation_restored | 至少一次恢复尝试取得过被接受的新曝光观测；不表示当前相机持续健康 |
| diagnosis.visual_confirmation_succeeded | 当前视觉确认状态为 confirmed_in_tray；不替代机械任务成功判定 |
| diagnosis.reason_code / explanation | 原始原因代码及简短中文说明；未知代码保留并提示查看明细 |
| diagnosis.cleanup_required | 尚未通过本控制器完成关闭，或上次关闭失败 |
| cleanup | 最近一次 close 的清理状态与错误；not_requested、succeeded 或 failed |
| camera | 导出时的相机会话、状态与采集错误 |
| result.recovery_attempts | 原有逐次恢复记录，包括故障、会话变化、恢复结果 |
| events | 原有事件与逐帧证据快照；不是完整驱动日志，也不补造缺失时间戳 |

修复了清理失败时可能仍把待映射尝试记成 observer_closed 的问题：现在记录 cleanup_failed，保存错误并继续抛出异常。后续成功重试关闭会更新 cleanup，但不抹去该次尝试曾经清理失败的结果。已经完成的视觉确认也不因清理失败被改写，两种结果并列展示。

## 使用

```python
from embodied_agent.recovery_report import recovery_report, write_recovery_report

# 在正常结束、停止或捕获异常后，先按既有生命周期约定尝试关闭。
# 即使 close 抛出异常，仍可在调用方的 finally 中导出诊断。
report = recovery_report(observer)
write_recovery_report("outputs/my-recovery-report.json", report)
```

导出使用 UTF-8 严格 JSON，拒绝 NaN/Infinity，并在序列化完成后才创建文件。已存在的路径不会覆盖，父目录须已存在；文件系统错误会直接抛出。此接口不是崩溃安全的持久化事务：写入期间磁盘故障可能留下部分文件。导出的原始错误文字可能包含驱动路径等信息，调用方应按实际用途处理。

## 场景报告与验证

已保存 [8 类确定性场景报告](evaluations/recovery-report-v1.json)。每个场景包含关闭前、关闭后的独立快照及实际捕获的清理异常：

1. 恢复观测，但稳定证据尚不足。
2. 恢复后重新累计稳定证据，视觉确认成功。
3. 再次故障且重启次数耗尽。
4. 剩余预算不足，未发起重启。
5. 重启操作失败。
6. 等待新映射期间停止。
7. 等待新映射期间关闭失败。
8. 视觉确认已成功，但随后清理失败。

运行 `make recovery-report-check` 会生成新的 outputs/recovery-report-v2.json；再次运行需用 `RECOVERY_REPORT_OUTPUT` 指定未占用路径。运行 `make recovery-report-test` 可复现新增的 8 项测试：结果区分、错误原因/计数、清理失败、快照隔离、JSON 往返及拒绝覆盖、无效数字拒绝、清理重试、视觉成功与清理失败并存。

本轮共 176 项相关测试通过：58 核心、31 接触、22 释放、13 线程采集、14 曝光、7 采集进程、8 进程恢复控制、23 恢复相关（含新增报告 8 项及既有 schema/validation 5 项）。Ruff 与差异检查通过。新报告场景均使用确定性会话桩，回归包含此前的真实子进程测试；没有接入真实相机或进行新的物理抓放实验。

后续已增加[离线审计与批量汇总](recovery-audit.md)：拒绝核心字段矛盾的报告，分别统计恢复、确认和清理，并避免把同一场景的关闭前后快照重复计数。
