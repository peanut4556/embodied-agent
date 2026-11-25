# 真实 CLI 子进程的端到端运行验收

日期：2026-11-25。

## 为什么增加整链验收

此前各模块已有针对性测试，但模块全部通过并不保证命令参数、退出码、文件格式及恢复步骤能串在一起。本次增加独立验收脚本，使用真实 Python 子进程和真实本地文件，贯穿任务执行到审计收尾。机器人仍是 MockRobot，不接硬件、不访问外部规划服务；这里只验证软件流程。

## 15 个验收步骤

| 步骤 | 检查内容 |
|---|---|
| initial | 创建有明确来源和序号的初始快照 |
| normal-task | CLI 执行四步 mock 任务，四步均确认完成 |
| duplicate-request | 同 ID 重复提交返回历史结果，预算文件字节不变 |
| request-conflict | 同 ID 改指令被拒绝，退出 3 |
| abrupt-process-exit | 子进程进入运行时后 os._exit(23)，退出 23 是预期故障 |
| pending-retry | 同一中断 ID 不再调用运行时，返回 operation_in_doubt |
| inspect-pending | CLI 查看待确认记录及检查摘要 |
| interrupted | 导出中断时的快照，保留动作暂停状态 |
| abandon-pending | 用检查摘要显式终止该请求，不修改预算文件 |
| abandoned-retry | 同 ID 返回 operation_abandoned，不执行 |
| final | 导出最终快照 |
| offline-verification | 独立 CLI 校验最终快照 |
| pending-to-abandoned-diff | 验证唯一状态变化为合法终止，前驱摘要匹配 |
| whole-lifecycle-diff | 从初始到最终明确识别两个新增操作 |
| final-inventory | 最终清单为 complete=1、abandoned=1、pending=0、invalid=0 |

中断后还会直接检查 locate/pick/place/verify 的暂停状态。拒绝重试使用只要被调用就抛异常的替身运行时，防止把“执行后再报错”误当作成功抑制。终止前后预算文件逐字节比较，验证该操作没有顺带解封。

## 使用与输出

```sh
PYTHONPATH=src:. .venv/bin/python scripts/validate_operational_lifecycle.py \
  --output outputs/evaluations/lifecycle-new

make operational-lifecycle-test
```

输出目录的父目录须存在，目标目录必须不存在；拒绝覆盖已有目录，也不允许指定现有业务目录继续运行。脚本在此新目录内初始化隔离用预算，生成三份快照、两个操作回执、锁文件和 report.json。它不会接管用户原有检查点。

每个子进程最多等待 20 秒，脚本检查预期退出码和关键返回字段。全部通过后控制台输出 passed=true、stages=15。步骤失败时抛异常并在 finally 中保存 passed=false 的报告及失败阶段记录；初始目录创建或初始化预算失败早于流程启动，不保证生成报告。

report.json 包含 Python 版本、各步骤 stdout/stderr、实际/预期退出码、脚本与 src/embodied_agent 顶层 Python 模块摘要，以及导出目录内所有既有 JSON 产物的 SHA-256（包含操作回执，不包含报告自身）。这些指纹用于定位本次验收版本，不是整个依赖环境的锁定或签名。

本次留存 [验收摘要](evaluations/operational-lifecycle-v1.json)，15 步均符合预期。原始产物位于本地 outputs/evaluations/operational-lifecycle-v2/，该目录按仓库规则不提交；摘要记录其相对位置、阶段结果和产物摘要。再次运行会因实时耗时等因素产生不同字节，验收依据是行为断言，不要求新报告逐字节等于历史报告。

## 回归与验收脚本自身的验证

新增三项测试：完整真实子进程链通过且每份产物摘要可复核；现有输出目录和文件不被覆盖；注入首步骤退出码错误后生成失败报告而不是误报通过。加上既有运行、预算、回执、快照和共享解析器回归，共 199 项相关测试通过，Ruff 检查通过。

Make 目标只在临时目录运行自动测试；显式验收命令才在指定新目录保留产物。没有改变运行时默认策略，没有新增训练、在线调参或实机性能结论。

## 说明与边界

该验收证明一个可重复的正常＋中断软件路径能通过 CLI 串通，不能代替所有故障组合、并发压力、真实网络重试、断电持久性或物理动作 exactly-once 测试。os._exit 是进程级故障，不是硬件断电。历史 ID 的保护范围仍局限于同一检查点协议，终止回执也不是物理结果证明。

这套脚本可作为后续修改的验收入口和面试演示材料：展示的不只是代码功能，还包括故障注入后能否保持暂停、是否拒绝重跑、审计记录是否能重现整个收尾过程。
