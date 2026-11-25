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
