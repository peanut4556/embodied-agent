# 预算重新校准：预览与显式应用

本轮把已有的独立验证重新校准接口接入运维 CLI，形成“隔离 → 新批次校准 → 独立验证 → 预览 → 按摘要应用”的流程。测试使用临时目录与模拟耗时，没有导入真实批次，也没有修改部署中的预算。

## 输入契约

请求是 UTF-8 JSON，严格字段如下。耗时单位为秒，表示动作执行加确认的总耗时。以下是结构示例，三个样本不足以通过默认校验，实际每批至少需要检查点的 `minimum_samples`（默认 20）个样本。

```json
{
  "schema": 1,
  "action": "verify",
  "calibration": {
    "batch_id": "verify-cal-001",
    "durations_s": [0.030, 0.031, 0.029]
  },
  "validation": {
    "batch_id": "verify-validation-001",
    "durations_s": [0.032, 0.031, 0.030]
  }
}
```

- 只接受 quarantine 模式，并要求目标动作已经处于隔离状态。
- 两批标识必须不同，且不能复用预算历史中的校准或验证批次标识。
- 样本必须是有限正数；每批最多 4096 个，文件最多 1 MiB。重复 JSON 字段、额外字段、未知版本及非标准数值均拒绝。
- 候选预算由校准批次最大耗时加原有 margin 冻结；独立验证批次每项都必须不超过该预算。验证数据不参与调高候选预算。
- 批次 ID 和摘要只能检查声明及文件一致性，无法证明两批实际独立、数据真实或具有代表性。

## 操作命令

在仓库根目录运行，替换路径为待维护检查点和新采集的请求文件：

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.budget_recalibration preview \
  --checkpoint /path/to/budget.json \
  --request /path/to/recalibration.json
```

成功输出 `ok: true` 和 `result`，包含目标动作、前后 revision、前后预算、前后隔离动作集合、校准与验证统计、两个原始文件 SHA-256，以及 `applied: false`。预览不会修改预算或请求文件；锁机制可能创建或保留 `.lock` 侧文件。

审核结果后，将预览中的两个摘要原样传入：

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.budget_recalibration apply \
  --checkpoint /path/to/budget.json \
  --request /path/to/recalibration.json \
  --expected-checkpoint-sha256 CHECKPOINT_SHA256_FROM_PREVIEW \
  --expected-request-sha256 REQUEST_SHA256_FROM_PREVIEW
```

应用时重新获取执行流程使用的同一把锁，重新校验并构建候选状态。任一摘要不一致则拒绝写入；文件仅改变空格也会使原始字节摘要变化。此时应重新预览和检查，不能直接把新摘要补入旧决定。摘要匹配且候选状态可持久化后才原子保存，成功输出 `applied: true`。

只恢复所选动作，其他隔离状态保持。revision 增加一，历史记录保存两个批次的标识、样本摘要和统计。使用旧预览重复应用会被拒绝；以后再次隔离也不能复用原批次。

## 错误与恢复

| 情况 | 行为 |
| --- | --- |
| 状态或请求变化 | 拒绝，通常为 `StaleRecalibration`；若请求已损坏，也可能先报告输入校验错误 |
| 有任务或维护进程持锁 | `CheckpointBusy`，不等待、不抢锁 |
| 样本不足、验证超预算、批次复用 | 拒绝，保留当前隔离 |
| 写入前失败 | 不发布候选；底层保存使用临时文件和原子替换 |
| 应用成功但终端未收到输出 | 重新读取预算和历史，不能假设失败后直接重放旧预览 |

成功退出 0；业务校验、锁和 I/O 错误退出 3，stderr 为 `ok: false`、`cause_type` 的 JSON，stdout 为空；缺少命令参数等 argparse 错误退出 2。

预算恢复不修改 pending/abandoned 回执，不自动重跑任务，也不证明此前动作的物理结果。pending 的处置仍走[回执检查与显式终止](operation-admin.md)。预览输出由调用方保存，当前没有独立维护审批日志。摘要不是身份认证，CLI 也不验证操作者权限。锁只约束同机合作写入者；不提供跨主机事务、恶意并发写保护或断电持久性保证。

## 验证

```sh
make budget-recalibration-cli-test
```

新增 10 项测试：只读预览和单动作恢复、检查点变化、请求变化、验证失败、严格 JSON、样本限制、保存前失败、重复应用与批次复用，以及真实 CLI 的成功/过期退出和锁冲突/缺参数。连同预算、执行恢复、回执、审计、进程退出矩阵等相关回归共 **212 项通过**。该数字是所选相关测试集合，不是全仓库测试总数。

保存失败测试在保存调用处注入异常，验证候选没有提前发布；不将其解释为覆盖所有文件系统故障时机。模拟耗时验证软件流程，不构成真实设备成功率或预算覆盖率结论。

## 响应丢失后的只读结果核对

如果 apply 已保存但终端断开，使用新命令核对当前检查点历史，无需再次应用：

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.budget_recalibration status \
  --checkpoint /path/to/budget.json \
  --request /path/to/recalibration.json
```

命令持同一把协作锁，严格读取检查点和请求，比较动作、两批标识、样本数量及两批样本摘要。它不会构建新预算、增加 revision、解除隔离或重放操作。

| status | 证据与处置含义 |
| --- | --- |
| `applied_current` | 当前历史中找到完整匹配，且该动作无后续校准、当前未隔离 |
| `applied_then_degraded` | 找到完整匹配，但动作当前再次隔离；历史应用成功不代表目前可执行 |
| `superseded` | 找到完整匹配，但该动作已有后续校准；查看 `later_action_revisions` 与当前隔离状态 |
| `batch_conflict` | 至少一个批次 ID 已使用，但动作、批次角色或样本内容不完整匹配；查看 `conflicting_revisions`，不要把它当成本请求已应用 |
| `not_found` | 当前检查点历史中未找到匹配或批次冲突；不能推出该请求从未在其他检查点、被回退前的状态或其他部署应用 |

输出包含 `matched_revision`、完整匹配记录 `matched_record`、`current_revision`、`currently_degraded`、`currently_blocked`、当前预算及漂移策略。后续其他动作的校准不会把本动作误判为 superseded。已被替代且再次隔离时，status 为 superseded，当前隔离字段仍为 true。

这里的匹配是样本内容和历史字段匹配，不是原请求文件字节匹配：JSON 缩进、字段顺序变化不影响核对；样本顺序和数字表示变化可能改变已保存的样本摘要。与 apply 的双原始字节摘要检查用途不同。status 不证明当时使用了哪份预览，也不认证部署来源。输出明确标记 `evidence_scope: current_checkpoint_history`。

成功读取和判定均退出 0，包含 conflict/not_found；调用方必须检查 `result.status`，不能把 `ok: true` 当作已应用。文件损坏、非法样本或锁冲突仍退出 3，绝不降级成 not_found。输入须有两批不同 ID，每批至少两个有限正数；实际应用时仍须满足检查点的 minimum_samples。只读核对不重新用当前配置检验旧批次的候选预算。

新增 10 项核对测试，包括一个真实子进程完成保存后直接退出 23、不输出响应，再由新 CLI 进程识别已应用且保持文件不变。该实验覆盖保存后响应丢失，不模拟断电或证明磁盘耐久性。

本轮连同原有校准、执行恢复、回执、快照与进程退出矩阵，**222 项相关测试通过**；新增测试同时检查查询前后预算与请求字节保持不变。

多个动作需要一起恢复时，可使用[多动作校准计划](budget-recalibration-plan.md)：全部校验成功后一次保存，避免逐项应用产生部分恢复。
