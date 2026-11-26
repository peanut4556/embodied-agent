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

响应丢失后可使用原来的单动作 `status`，分别传入计划内原始请求文件，核对每个动作历史。现在也可使用下述整体 status 命令，在同一锁内核对所有请求。计划尚无独立持久化回执，无法从历史匹配证明它们当时一定由同一计划提交。遇到检查点回退或更换来源，not_found 不能证明从未应用。

校准不会改变 pending/abandoned 操作回执，不会自动启动任务或重放不确定动作。是否运行新任务仍取决于当前隔离状态、运行预算及操作流程。

## 回归验证

```sh
make budget-recalibration-plan-test
```

新增 11 项测试覆盖：整体预览无写入、一次保存与保留无关隔离、第二项验证失败不发布第一项、重复动作/跨角色批次复用、输入数量与严格 JSON、检查点和计划变化、保存前异常、重复应用/历史批次复用、目标和策略限制、真实 CLI 成功/失败/锁冲突，以及真实子进程保存前后退出。

相关回归合计 **233 项通过**，涵盖原单动作校准、结果核对、执行恢复、操作回执、快照审计与故障矩阵。使用临时模拟数据，没有导入真实批次或调整部署中的预算。


## 整份计划的只读核对

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.budget_recalibration_plan status \
  --checkpoint /path/to/budget.json \
  --plan /path/to/recalibration-plan.json
```

status 获取同一把执行协作锁，只加载一次检查点，然后核对计划的所有请求。所有动作的结论来自同一份已验证预算状态，避免逐项查询期间发生维护更新而混合不同版本。不会重新校准、解除隔离、保存检查点或重放任务；可能创建或保留锁侧文件。

| 汇总 status | 判定 |
| --- | --- |
| all_matched | 所有请求都在当前历史中找到完整内容匹配 |
| partially_matched | 部分匹配，其余未找到，且没有批次冲突 |
| conflict | 至少一个请求的批次 ID 已用但内容不匹配；优先于部分匹配 |
| not_found | 所有请求均未找到匹配或批次冲突 |

输出包含 current_revision、原始计划字节摘要 plan_sha256、request_count、matched_count、各单动作状态的 counts，以及按计划顺序排列的 results。currently_blocked_actions 仅列出计划内当前被预算隔离阻止的动作，不是完整系统就绪检查。其他未列入计划的隔离动作仍可能阻止任务。

results 使用已有单动作核对规则，区分 applied_current、applied_then_degraded、superseded、batch_conflict、not_found。因此 all_matched 可以同时包含已再次隔离和被后续校准替代的动作。后续其他动作的校准不影响本动作的匹配。内嵌请求不是独立文件，结果不虚构逐请求文件摘要；顶层只返回实际计划文件摘要。

输出始终包含 `evidence_scope: current_checkpoint_history` 和 `atomic_application_verified: false`。即使历史记录连续、全部匹配，也可能来自分别执行的单动作命令；不能从该查询证明同一计划曾原子提交。partially_matched 也不能直接认定批量保存发生了部分提交，可能只是之前只维护过部分动作。not_found 不能排除状态回退、来源更换或其他部署曾经应用。

调整计划请求顺序或 JSON 排版后，历史内容仍可匹配，但 plan_sha256 会变化。status 的历史核对与 apply 的原始字节摘要检查用途不同，不能据此绕过重新预览。

查询成功退出 0，包括 conflict 和 not_found；自动化必须检查 result.status 和各动作当前状态，不能仅凭 ok:true 决定重跑或恢复。非法计划、损坏检查点、锁冲突退出 3，不伪装成历史未找到。查询沿用计划数量、文件大小和批次唯一性限制。

本轮新增 8 项测试：全部/未找到、部分匹配、冲突优先、再次隔离与后续替代、顺序/格式变化、CLI 与锁冲突、损坏输入，以及真实子进程批量保存后退出 23，再由新 CLI 核对全部匹配且不改写文件。相关回归合计 **241 项通过**。该进程退出实验不代表断电耐久性验证。

## 一次诊断全部动作

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.budget_recalibration_plan diagnose \
  --checkpoint /path/to/budget.json \
  --plan /path/to/recalibration-plan.json
```

preview 和 apply 仍在遇到失败时拒绝整份计划。diagnose 则用于准备阶段：在同一锁内读取预算，逐动作收集可判断的问题，让操作者一次看到多个待修项。推荐流程为 diagnose → 修复输入或重新采集 → preview → 审核整体变化 → apply → 必要时 status 核对。

返回 `ready_for_preview`、`eligible_count`、`request_count`、`minimum_samples`、`current_revision`、`plan_sha256`、`combined_error` 和按输入顺序排列的 results。每项包含 action、eligible、reasons、两批样本数量及 candidate_record。`applied` 始终为 false。

| reasons 中的稳定代码 | 含义 |
| --- | --- |
| quarantine_policy_required | 当前检查点不是 quarantine 策略 |
| action_not_degraded | 目标动作当前未隔离，不满足重新校准前提 |
| insufficient_calibration_samples | 校准批次数量不足 |
| insufficient_validation_samples | 独立验证批次数量不足 |
| historical_batch_conflict | 校准或验证批次 ID 已在历史中使用 |
| validation_exceeds_candidate | 验证样本超过校准最大值加 margin 的候选预算 |
| candidate_not_persistable | 前置条件通过，但单动作候选未通过持久化约束，例如容量超限 |

同一项可以同时报告多个原因。例如校准和验证批次都只有两个样本、且验证耗时超预算，会同时显示三项问题。候选构建与持久化检查仅在该动作前置检查通过后执行，因此 reasons 是本次能判定的问题集合，不承诺穷尽所有隐藏问题。

所有动作分别合格后，诊断会按计划顺序构建完整候选，再次检查组合约束。若各动作单独可加入预算，但一起加入后超过 128 个估计项，则 eligible_count 可能等于 request_count，ready_for_preview 仍为 false，combined_error 为 `combined_candidate_not_persistable`。不能仅用 eligible_count 决定是否继续。

candidate_record 仅是从同一原始预算独立构建的单动作候选记录。多个候选可能显示相同 revision，它们不是已保存记录，也不是完整计划的最终历史序号；以随后 preview 的 changes 为整体预览依据。诊断不返回可用于 apply 的检查点摘要，不能替代预览。诊断成功后，状态仍可能变化；apply 继续重新验证并检查双摘要。

### 错误分层

- 退出 0：诊断完成且 ready_for_preview 为 true，不代表已应用或设备就绪。
- 退出 1：诊断完成，但存在动作或组合问题。stdout 为 `ok: true` 的结构化结果，stderr 为空。
- 退出 3：无法可靠诊断，例如计划结构非法、重复动作/批次、非法数值、损坏检查点或锁冲突；stderr 为 `ok: false` 与 cause_type，stdout 为空。
- 退出 2：命令参数不完整等 argparse 错误。

格式非法不会被拆成猜测性的逐项结论，仍沿用严格 JSON、总文件大小、动作数量、批次数量与数值限制。diagnose 不改写预算或计划、不解除隔离、不自动扩大预算，也不绕过独立批次要求；可能创建锁侧文件。它没有设备健康检查或真实数据代表性判断。

新增 8 项测试覆盖只读合格诊断、多动作多原因汇总、策略不符、历史批次冲突、单动作容量失败、真实组合容量失败、CLI 的 0/1 退出码及格式/锁错误。连同维护、恢复、审计与运行时相关回归共 **249 项通过**。测试使用临时模拟状态，未操作真实部署预算。
