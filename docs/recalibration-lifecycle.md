# 校准维护整链验收

本验收将分散的诊断、批量预览、应用和结果核对串成真实 CLI 子进程流程。全部状态在新建的输出目录内生成，数据为合成耗时，不调用机械臂、不修改部署检查点，也不验证真实设备成功率。

## 运行和产物

```sh
PYTHONPATH=src:. .venv/bin/python scripts/validate_recalibration_lifecycle.py \
  --output outputs/evaluations/recalibration-lifecycle-new
```

输出目录必须不存在，父目录必须存在；已有目录直接拒绝，避免覆盖旧证据。每个子进程超时 20 秒，使用当前 Python 解释器和仓库源码。成功摘要为 passed:true、stages:11。

报告 report.json 保存每步名称、预期/实际退出码、stdout、stderr、Python 版本及通过状态。还包含顶层运行源码的 SHA-256 和输出 JSON 产物摘要。输出保存初始、模拟并发更新后、成功应用后、再次漂移后的四份检查点，以及非法计划和当前修复计划。报告本身不递归包含自己的摘要。失败时尽可能保存已有阶段和异常信息，随后让错误传播，避免把中断验收记为成功；输出目录创建失败发生在报告创建之前。

摘要用于复核文件是否变化，不是签名或可信来源认证。源码摘要范围为验收脚本和 src/embodied_agent 下顶层 Python 文件，不等于完整依赖环境锁定。后续源码更新可能使报告中的摘要与当前文件不同，应重新验收，不能直接覆盖旧报告。

## 11 个阶段及验证目标

| 阶段 | 预期 |
| --- | --- |
| diagnose-invalid | 退出 1，同时发现抓取样本不足和核验超预算 |
| preview-invalid | 退出 3，检查点字节保持不变 |
| diagnose-fixed | 修复计划后退出 0，允许进入预览 |
| preview-fixed | 获取第一份预览双摘要 |
| reject-stale-preview | 模拟另一状态更新后应用旧预览，退出 3 且不写入 |
| preview-current | 基于当前状态重新预览 |
| apply-current | 两动作一起恢复，revision 从 0 到 2 |
| status-applied | 全部历史匹配，两个动作都是当前校准；无关隔离保留 |
| reject-repeat-apply | 重复旧应用退出 3，文件不变 |
| status-after-drift | 抓取耗时再次超预算，历史仍匹配但当前抓取被隔离 |
| diagnose-reused-batches | 退出 1，两批历史复用问题明确报告，不自动重新恢复 |

测试不仅比较退出码，还检查业务结果和预算字节不变性。修复输入和注入后续漂移由验收脚本完成，诊断与维护调用均通过真实 CLI。中间状态变化是确定性顺序注入，不声称模拟了真实并发竞争；实际锁冲突与进程退出边界由已有专项测试覆盖。

## 本轮结果

本地验收 11 个阶段全部通过。原始报告及状态文件位于 outputs/evaluations/recalibration-lifecycle-v1；随仓库保存的报告副本为 [验收报告](evaluations/recalibration-lifecycle-v1.json)。原始目录通常被 Git 忽略，报告中的产物摘要需要配合本地原始文件核对；复制报告不等于归档了全部原始产物。

新增三项测试验证整链、产物摘要和报告一致性，已有目录不覆盖，以及子进程失败时保存失败报告。连同维护、执行恢复、回执、审计及运行时相关回归，共 **252 项通过**。

```sh
make recalibration-lifecycle-test
```

这是一份可复现的软件维护验收，不证明批次真实独立、预算具有统计覆盖保证，也不证明设备物理状态。all_matched 仍只是当前历史内容匹配；不能据此认定设备当前可执行或证明这些记录由同一计划提交。实际维护步骤参见[批量校准计划](budget-recalibration-plan.md)。

## 离线证据复核

现在可对原始输出目录独立复核，无需重新执行校准：

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.recalibration_evidence \
  --directory outputs/evaluations/recalibration-lifecycle-v1

# 可选：同时比较报告记录的源码与当前工作树
PYTHONPATH=src .venv/bin/python -m embodied_agent.recalibration_evidence \
  --directory outputs/evaluations/recalibration-lifecycle-v1 \
  --source-root .
```

工具严格解析 report.json，检查完整的 11 阶段顺序、预期和实际退出码，并检查七个必需产物的 SHA-256。产物缺失返回 missing，内容变化返回 changed，不把缺失视为成功。未提供 source-root 时 source_checked 为 false；提供后逐项核对报告中声明的源码。当前源码有变化只表示与该次验收不同，不等于当时验收无效。

输出 scope 固定为 recorded_file_integrity_and_stage_exit_codes，authenticated 为 false。退出 0 表示该范围内复核通过；退出 1 表示存在缺失、变化、阶段退出码不符或报告 passed 为 false；退出 3 表示报告/路径非法或 I/O 错误，stderr 为结构化错误。缺少参数退出 2。

复核不会重新解释 stdout 中每个业务断言，不会重跑任务，也不认证报告生成者。修改报告并重算摘要可以伪造一致性；没有独立签名或可信外部基准。源码核对仅覆盖报告声明的允许路径，不证明依赖环境完整。失败或中断报告不符合完整成功报告结构时会拒绝，不作为完成验收证据。

### 读取边界

- 报告和每个被检查文件最多 1 MiB；源码清单最多 256 项，产物清单必须恰好为七个已定义文件，最大内容读取量为 264 MiB（含报告）。
- 路径必须为相对 POSIX 路径，拒绝绝对路径、父目录跳转、空段和反斜杠。
- 源码只允许验收脚本及 src/embodied_agent 下顶层 Python 文件；拒绝路径中的符号链接、非普通文件以及作为符号链接的报告。
- 显式指定的根目录解析后作为信任边界。复核不持运行锁，应对停止写入的归档执行；不提供恶意并发改写下的文件系统安全保证，也不会扫描额外未列入清单的文件。
- 只读复核不修改原始报告或产物；仓库中的单份报告副本缺少配套原始文件，不能独立通过完整产物核对。

本轮新增七项测试覆盖正常与源码核对、缺失/变化、阶段与清单不完整、路径跳转/符号链接、超大文件/重复 JSON、缺失源码及 CLI 退出码。相关回归共 **259 项通过**，并对本地上一轮原始验收目录完成了产物与声明源码复核。
