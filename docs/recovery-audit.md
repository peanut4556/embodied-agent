# 恢复报告的离线审计与批量汇总

2026-10-30。新增 `recovery_audit` 与命令行入口，在不访问相机、不修改输入报告的前提下检查核心字段自洽性并汇总。它不校验数据真实性，不重新计算视觉稳定证据，也不是完整 JSON Schema 验证器。

## 检查与计数

支持单个 v1 恢复快照，以及 `check_recovery_reports.py` 导出的 v1 确定性场景集合。未知版本和集合类型被拒绝，避免把不兼容字段默认为成功。

- 核对诊断标志与状态、原因、恢复尝试记录一致；严格区分布尔值与整数。
- 核对重启计数、次数上限、连续编号、故障类型及尝试时间顺序。声称恢复观测的尝试必须有新会话及期限内的启动完成时间。
- 核对清理状态与错误，成功清理必须对应 closed 相机状态。
- 场景集合同时检查关闭前后的快照，检查关闭不改变原始期限、会话和次数，以及终态变化与清理异常是否一致。只将关闭后快照计入总数，避免同一场景重复计数。
- 无效场景从统计中排除并单独记录具体错误，其他有效场景继续处理。有效条数与拒绝条数分别报告；拒绝条目可能是整个文件、重复输入或单个场景，不应将它们合并解释为任务成功率。
- 同一解析后的文件路径重复输入会被拒绝；不同路径下复制的同一实验无法据此识别，调用方仍需保证输入批次独立。
- 拒绝重复 JSON 键、NaN/Infinity、指数溢出和损坏 JSON。汇总文件不覆盖已有文件。

输出包括 accepted/rejected 明细、status_counts、cleanup_counts 与分项 counts。其中 observation_restored 表示“曾取得有效新观测”，visual_confirmation_succeeded 表示视觉确认成功，cleanup_required 表示尚需清理，restart_count 是尝试总次数。这些计数的语义不同，不合并为单一成功率。单个仍在运行的快照可以被接受，其 pending 状态会单独统计。

逐帧 events 只检查基本容器结构；审计不能从当前报告重建完整的原始坐标与视觉稳定过程。通过审计仅说明所检查的字段彼此一致。

## 使用

```sh
make recovery-audit-test
make recovery-audit-check
PYTHONPATH=src:. .venv/bin/python scripts/audit_recovery_reports.py \
  outputs/run-a.json outputs/run-b.json --output outputs/recovery-summary.json
```

默认输入为 docs/evaluations/recovery-report-v1.json，默认输出为 outputs/recovery-audit-v2.json，可通过 RECOVERY_AUDIT_INPUT / RECOVERY_AUDIT_OUTPUT 更改。输出目录需存在，重复执行需换用新输出路径。

有任何拒绝项时，CLI 仍导出明细并返回退出码 1；全部通过返回 0。输出写入失败也以非零退出，不覆盖旧报告。

## 本轮结果

[保存的审计汇总](evaluations/recovery-audit-v1.json) 接受 8 个确定性场景、拒绝 0 个：

| 指标 | 数量 |
| --- | ---: |
| 曾恢复有效观测 | 3 |
| 视觉确认成功 | 2 |
| 尚需清理 | 2 |
| 清理成功 / 失败 | 6 / 2 |
| 重启尝试总数 | 7 |

新增 8 项测试覆盖字段矛盾、时间与会话错误、无效场景排除、重复/损坏/非有限 JSON、重复输入、关闭前后不一致、CLI 退出码及拒绝覆盖。相关回归共 184 项通过（58 核心、31 接触、22 释放、13 线程采集、14 曝光、7 采集进程、8 进程恢复、31 恢复相关）。Ruff 与差异检查通过。

这些是合成场景的诊断计数，不代表实际设备的可靠性或抓放成功率。本轮未连接真实设备、未新增训练或物理实验。

后续已增加[曝光证据离线重放](recovery-replay.md)：新报告保留原始输入和映射，审计可复算逐帧确认；旧报告明确标为无法重放。以上仅字段自洽的描述对应最初版本，当前汇总额外包含 replay_counts。
