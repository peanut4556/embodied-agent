# 操作回执的检查与显式终止

日期：2026-11-24。

## 检查接口

新增 inspect_operation(checkpoint, operation_id)，持有与执行相同的非阻塞协作锁后，读取并严格校验回执和当前预算。结果包含原回执、规范化回执摘要、预算有效性/版本/策略/退化集合、historical_result_available 和 physical_outcome_verified=false。

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.operation_admin inspect \
  --checkpoint budget-state.json --operation-id pick-request-001
```

有效回执配损坏预算仍可报告：receipt 正常返回，budget.valid=false 并带异常类型；损坏或缺失回执则整个检查失败，不猜测操作状态。inspect 不修改预算和回执，但为协调并发可能创建固定 .lock 侧文件，因此“只读”指业务状态只读。持锁者正在运行时立即报告冲突，不等待也不抢锁。

historical_result_available=true 仅表示存在已完成的历史 TaskResult，不允许执行动作，也不证明物体当前状态。操作 pending、预算已提交可以同时存在，例如预算提交成功但写回执失败；两份状态分开呈现，不把这种组合强行解释为成功或未执行。

## 显式终止待确认操作

新增 abandon_operation 只接受 pending，写入不可自动重放的 abandoned 终态。必须提供 expected_receipt_sha256，且在锁内检查它与当前回执一致，避免检查后回执变化仍被旧决定覆盖。同时记录 reason、evidence_ref、actor 和 previous_receipt_sha256。

```python
from embodied_agent.operation_admin import inspect_operation, abandon_operation
review = inspect_operation(path, operation_id)
# 操作者核对外部设备/任务记录后，明确决定不再继续此请求。
closed = abandon_operation(
    path, operation_id,
    expected_receipt_sha256=review["receipt_sha256"],
    reason="request cancelled after external review",
    evidence_ref="review-20261124-001",
    actor="operator",
)
```

命令行也提供同样参数：

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.operation_admin abandon \
  --checkpoint budget-state.json --operation-id pick-request-001 \
  --expected-receipt-sha256 <检查结果中的摘要> \
  --reason "request cancelled" --evidence-ref review-001 --actor operator
```

这是显式修改业务回执的操作，应只在操作者已经决定终止该请求时调用。此次开发仅在临时测试目录验证，不处理用户现有任务记录。

成功返回 ok=true 和结果，退出 0；输入/文件/锁错误向 stderr 输出 ok=false 与异常类型，退出 3；缺少命令参数由 argparse 退出 2。错误输出不泄露原始路径或异常内容。

## 不改变什么

终止记录不写预算文件，不解除动作隔离，不生成成功结果，不取消正在执行的物理动作，也不删除原请求摘要。后续同 ID 同请求返回 operation_abandoned，而非重新调用运行时；同 ID 不同参数仍返回 operation_conflict。历史 complete 不允许终止覆盖；第二次终止也拒绝，不静默替换审计信息。

reason、evidence_ref、actor 是调用方声明，不是身份认证或外部证据验证。当前没有查询外部记录或设备的能力；不能用终止记录证明“动作从未发生”。异常状态的物理结果仍可能未知，不能通过换新 ID 重试来绕过核对。

## 格式与兼容

抽出统一 read_receipt，执行路径和管理路径使用同一套校验。schema 1 保留 pending/complete；schema 2 专用于 abandoned，result 必须为 null，resolution 四字段必须完整、标签有界且摘要格式正确。SHA-256 校验覆盖完整规范化回执。旧客户端遇到 schema 2 会拒绝，不应按缺失记录处理；升级客户端才能识别 operation_abandoned。

本次也统一拒绝 pending 带非空 result、无效 schema、终态带伪造 TaskResult 等不一致结构。文件写入继续使用同目录临时文件和原子替换。旧 pending 内容的请求摘要与其完整摘要保留在终态记录中，但没有另存不可变原文件；这不是追加式防篡改审计日志。

## 验证与限制

新增 10 项测试覆盖检查不改业务文件、预算损坏时仍显示有效回执、终止不解封/不重执行、旧摘要拒绝、二次终止拒绝、complete 不可覆盖、损坏回执拒绝、写入失败保持 pending、终态结构校验、真实 CLI 检查/锁冲突和终止流程。相关回归共 143 项通过。

仍依赖可信本地文件、单主机合作锁与操作者真实性。摘要不是签名，不能阻止有写权限者改文件并重算摘要；不提供断电事务、跨机器锁或物理动作 exactly-once。回执不能自动清理，否则会失去同 ID 的重复抑制证据。

```sh
make operation-admin-test
```
