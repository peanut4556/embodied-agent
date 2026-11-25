# 操作提交边界的进程退出矩阵

日期：2026-11-25。

## 验证范围

此前整链验收只有一个执行中退出案例。本次增加七个不同提交边界，每个案例先启动真实 Python 子进程，到指定位置 os._exit(23)；再启动另一个子进程提交同一 ID、同一参数。通过真正的 run_operation、检查点保存和 AgentRuntime 流程，检查回执状态、预算暂停集合和 mock 动作日志。

动作替身只执行单步 verify；execute 会向隔离目录 effects.jsonl 写一条事件并 fsync，再按配置退出。日志计数用于检测实际 Python execute 调用次数，不是机械动作次数。故障注入通过测试脚本局部 patch 写入函数实现，没有在生产路径加入可触发退出的开关。

## 七个案例结果

| 退出位置 | 重试前回执 | 重试前动作数 | 重试行为 | 最终动作数 |
|---|---|---:|---|---:|
| 登记前 | absent | 0 | 正常首次执行 | 1 |
| 登记后 | pending | 0 | operation_in_doubt | 0 |
| 预写暂停后 | pending | 0 | operation_in_doubt | 0 |
| 动作事件写入后 | pending | 1 | operation_in_doubt | 1 |
| 最终预算保存后 | pending | 1 | operation_in_doubt | 1 |
| 最终回执写入前 | pending | 1 | operation_in_doubt | 1 |
| 最终回执写入后 | complete | 1 | 返回历史成功结果 | 1 |

七个首次子进程均在预期位置退出 23，七次重试均正常返回协议结果；14 个子进程构成一次矩阵验收。没有任何案例出现第二次 execute 调用。登记后的零动作案例也被保守阻止，这是不确定状态牺牲可用性的实际体现，不应称为所有故障都能自动恢复。

暂停状态与重复抑制分工不同：预写后/动作后退出时预算保持暂停；最终预算已保存但回执尚未完成时，预算可以已经清除临时暂停，仍由 pending 回执阻止同 ID 重跑。因此不能只看预算判断操作可否重试，也不能换 ID 绕过不确定操作核对。

## 运行和证据

```sh
PYTHONPATH=src:. .venv/bin/python scripts/validate_crash_boundaries.py \
  --output outputs/evaluations/crash-boundaries-new

make crash-boundaries-test
```

输出必须是新目录，每个案例有独立预算、回执和动作日志，脚本不接管现有检查点。每个子进程最多等待 20 秒。--child/--fault 是内部工作进程参数，日常只使用 --output 入口。

report.json 保存每例退出码、重试结果、前后动作数、回执状态、重试前退化集合、Python 版本，以及脚本/关键模块和实际 JSON/JSONL 产物摘要。失败也写 passed=false 的报告并抛异常；创建输出目录失败则不改旧目录。原始文件保留在本地 outputs/evaluations/crash-boundaries-v1/，提交的 [验收记录](evaluations/crash-boundaries-v1.json) 保存对应摘要与逐例结果。

## 回归和边界

新增三项测试：完整矩阵及所有产物摘要复核；现有输出目录不可覆盖；注入错误退出码后明确记录失败。与已有功能、整链验收和共享解析器回归共 202 项相关测试通过，Ruff 检查通过。没有改动默认预算策略或实际执行协议。

测试覆盖的是写入函数返回前后的命名边界，不包含写入内部每条指令、任意时刻抢占或所有文件系统故障。os._exit 是软件进程退出，不是断电；动作日志 fsync 也不是设备独立传感器见证。只有同一检查点、相同 ID 和一致参数的合作调用方受保护，不能外推为物理 exactly-once、跨主机事务或换 ID 后也不重复。

这些结果可作为提交前验收材料：不仅展示正常功能，还展示在不同“结果已发生但记录尚未提交”的时点，系统如何保持不确定、拒绝自动重放，并保留后续人工核对所需证据。
