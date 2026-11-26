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
