# 多动作校准计划：全部验证后一次保存

单动作 CLI 适合独立维护；如果一次维护需要同时恢复抓取和核验，逐项应用会在后项失败时留下部分恢复状态。本接口先在内存中验证整份计划，全部通过后才保存一次检查点。旧单动作 preview/apply/status 保持可用，二者共享输入和候选预算校验。

## 输入与验证

计划采用严格 JSON，顶层只有 `schema: 1` 和 `requests`。requests 是 1 至 16 个完整的单动作请求，按列表顺序校准。每个请求包含 schema、action、calibration、validation，格式见[单动作契约](budget-recalibration-cli.md)。

下面将两份已准备的单动作请求组合为计划，仅创建新文件，不读取或修改检查点：

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
requests = [json.loads(Path(name).read_text()) for name in
            ('pick-request.json', 'verify-request.json')]
with Path('recalibration-plan.json').open('x') as output:
    json.dump({'schema': 1, 'requests': requests}, output, indent=2)
PY
```

CLI 会对最终计划重新进行严格解析和验证：

- 整份文件最多 1 MiB；每批最多 4096 个有限正数，每批数量必须达到检查点 minimum_samples。
- 同一动作只能出现一次；所有动作的校准和验证批次 ID 必须全局唯一，不能跨角色复用，也不能复用已有历史中的 ID。
- 所有目标动作必须已经隔离，检查点必须为 quarantine 策略。
- 每个候选预算由该动作校准样本最大值加原 margin 确定；验证批次不能参与调高预算，任一验证样本超预算则整份计划拒绝。
- 每一步候选都通过持久化结构校验，包括历史容量限制；即使第一步可保存、第二步超限，也不会发布第一步。
- 不接受重复 JSON 字段、额外字段、未知版本、布尔数值、非有限数值或空计划。

批次独立性、数据真实性和采集覆盖范围仍由采集流程保证，代码不能仅凭不同 ID 证明它们。

## 预览和应用

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.budget_recalibration_plan preview \
  --checkpoint /path/to/budget.json \
  --plan /path/to/recalibration-plan.json
```

返回 `ok: true`、`result`，其中包含：

| 字段 | 含义 |
| --- | --- |
| checkpoint_sha256 / plan_sha256 | 被预览的两个文件原始字节摘要 |
| before_revision / after_revision | 整份计划应用前后的历史版本 |
| before_degraded / after_degraded | 整体隔离状态变化 |
| changes | 按输入顺序列出每个动作的新旧预算及校准历史记录 |
| applied | 预览为 false，保存完成后为 true |

审核整体变化后，将两个摘要传给 apply：

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.budget_recalibration_plan apply \
  --checkpoint /path/to/budget.json \
  --plan /path/to/recalibration-plan.json \
  --expected-checkpoint-sha256 CHECKPOINT_SHA256_FROM_PREVIEW \
  --expected-plan-sha256 PLAN_SHA256_FROM_PREVIEW
```

预览和应用各自持同一把执行协作锁；应用重新验证整份计划，摘要不匹配则拒绝。调整请求顺序或 JSON 格式也会使计划原始字节摘要变化，需要重新预览。输入若已变得非法，可能先报告输入校验错误。不能使用过期预览覆盖中间发生的执行或维护更新。

候选全部验证成功后只调用一次 save_budget，通过原有临时文件和原子替换发布。两个动作恢复时 revision 增加 2，并各留一条历史，不额外增加“计划版本”。未列入计划的动作及其隔离状态保持。任意前置校验失败都不发布部分候选。

成功退出 0；锁冲突、过期预览、校验和 I/O 错误退出 3，stderr 返回 ok:false 与 cause_type，stdout 为空；缺少参数退出 2。锁冲突不自动等待或抢锁。不会自动重试计划。

## 保存边界与结果核对

保存之前进程退出，仍是旧预算；保存函数完成后进程退出，当前检查点已经包含整份计划，即使终端没有收到成功响应。测试在这两个调用边界分别直接退出真实子进程，检查文件始终为旧版本或完整的新版本。

该实验没有覆盖文件系统内部全部故障时机，也不提供断电耐久性或跨主机事务保证。这里的一次发布仅针对预算检查点，不涉及回执文件与物理设备的事务。预览可能创建锁侧文件；没有维护权限认证，摘要不是签名。

响应丢失后可使用原来的单动作 `status`，分别传入计划内原始请求文件，核对每个动作历史。计划本身尚无独立持久化回执或整体 status 命令，无法从逐项匹配证明它们当时一定由同一计划提交。遇到检查点回退或更换来源，not_found 不能证明从未应用。

校准不会改变 pending/abandoned 操作回执，不会自动启动任务或重放不确定动作。是否运行新任务仍取决于当前隔离状态、运行预算及操作流程。

## 回归验证

```sh
make budget-recalibration-plan-test
```

新增 11 项测试覆盖：整体预览无写入、一次保存与保留无关隔离、第二项验证失败不发布第一项、重复动作/跨角色批次复用、输入数量与严格 JSON、检查点和计划变化、保存前异常、重复应用/历史批次复用、目标和策略限制、真实 CLI 成功/失败/锁冲突，以及真实子进程保存前后退出。

相关回归合计 **233 项通过**，涵盖原单动作校准、结果核对、执行恢复、操作回执、快照审计与故障矩阵。使用临时模拟数据，没有导入真实批次或调整部署中的预算。
