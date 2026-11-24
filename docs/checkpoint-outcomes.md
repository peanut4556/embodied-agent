# 检查点执行的结构化异常合同

日期：2026-11-24。

## 两种调用接口

run_checkpointed 保留原接口：返回 TaskResult 或抛原始异常。新增 run_checkpointed_report 使用相同执行流程，捕获普通 Exception，返回结构化控制结果；KeyboardInterrupt、SystemExit 等 BaseException 不吞掉，已经预写的暂停状态仍保留。

```python
from embodied_agent.checkpointed_runtime import run_checkpointed_report
outcome = run_checkpointed_report(runtime, instruction, path, task_timeout_s=2)
if outcome["error"] is not None:
    # 记录错误阶段和状态，不能自动重放动作。
    handle_error(outcome)
else:
    handle_task_result(outcome["task_result"])
```

外层固定字段为 success、task_result、checkpoint、error。正常完成时 task_result 是完整 TaskResult 字典，error 为 null；任务判定失败仍属于正常返回，此时 success=false，但 error=null。基础设施异常时 task_result=null，不伪造完成步数或部分成功结果。

## 阶段与检查点状态

| phase | 意义 | execution_started |
|---|---|---|
| validate | 参数验证 | false |
| lock | 获取协作锁 | false |
| load | 加载并验证文件 | false |
| policy | 检查 quarantine 策略 | false |
| prepare | 写入保守暂停快照 | false |
| execute | 调用运行时 | true |
| commit | 写入实际最终状态 | true |
| unlock | 释放锁 | true |
| complete | 执行流程完成 | true |

execution_started 表示运行时已被调用，不证明 execute 已被调用或物理动作发生。异常中的 action_outcome_unknown 使用同样的保守边界：一旦进入运行时，调用方不能凭本接口推断动作未发生。即使运行时实际在规划阶段就失败，也按可能执行处理。

checkpoint_state 为 unchanged（本流程尚未尝试写）、not_confirmed（正在写或写入失败，不能确认最终状态）、prepared（保守快照已保存）、committed（最终状态已保存）。加载失败时 unchanged 不代表文件有效；最终写入失败不能声称磁盘必然还是 prepared。解锁失败时可出现 committed 加错误，表示保存已完成但流程尚未正常结束。

错误对象包含 code、cause_type、action_outcome_unknown。锁冲突固定 checkpoint_busy，其余按阶段返回 checkpoint_validate_failed / checkpoint_lock_failed / checkpoint_load_failed / checkpoint_policy_failed / checkpoint_prepare_failed / checkpoint_execute_failed / checkpoint_commit_failed / checkpoint_unlock_failed。不包含原始异常文本、路径或 traceback；Python 调试调用仍可使用原始异常接口。

## CLI 合同

带 --budget-checkpoint 时使用结构化接口。--json 正常任务结果保持原有顶层 success/message/completed_steps/confirmation_reports，并追加 checkpoint；基础设施错误输出上述错误信封，task_result=null。外部读取方先检查 error 是否存在且非空，再读取任务字段。

退出码：成功任务 0，已返回的失败任务 1，命令行参数解析错误 2，检查点执行流程错误 3。文本模式也返回 3 并列出错误码、execution_started 和 checkpoint_state。不再因这些普通执行异常输出 traceback。运行时构造阶段（如非法确认参数）和非检查点路径的异常处理未纳入本合同；不要据此假定所有程序异常均输出 JSON。

示例基础设施错误：

```json
{"success":false,"task_result":null,"checkpoint":{"phase":"commit","execution_started":true,"checkpoint_state":"not_confirmed"},"error":{"code":"checkpoint_commit_failed","cause_type":"OSError","action_outcome_unknown":true}}
```

锁冲突说明本次未进入运行时，可等待现有持锁者结束后重新评估；不要删除锁文件。执行或提交失败时应保留暂停，检查设备和检查点，再决定恢复；本接口不发出“可自动重试”的承诺。即使错误发生在执行前，也要先解决原有故障，不能盲目循环。

## 测试与边界

新增 10 项测试覆盖正常成功/任务失败、缺失与损坏文件、预写失败、运行异常、提交失败、KeyboardInterrupt、真实 CLI 锁冲突、JSON/文本错误、参数与策略拒绝、解锁失败。验证保存失败不输出成功、敏感异常文本不进入结果、预写暂停保持、锁冲突不写文件。相关回归共 123 项通过。

这完善了上位机可读的错误语义，不增加实机安全能力；仍受前述本地协作锁、可信路径、非断电持久性和物理执行结果不确定等限制。检查点状态描述本软件已完成的操作，不是对磁盘硬件或设备状态的独立见证。
