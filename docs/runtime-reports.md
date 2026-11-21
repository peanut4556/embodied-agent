# 结构化动作确认诊断

日期：2026-11-21。

TaskResult 新增默认空列表 confirmation_reports，旧三个位置参数继续可用。基础运行链路为每个进入确认阶段的步骤记录 action、step、reads、stale_reads、negative_reads、duplicate_reads、streak_resets、gap_resets、final_streak、required_samples、required_span_s、elapsed_s 和结束 status。多样本模式还记录当前段 observed_span_s。

reads 是成功返回的 observe 次数，包含随后因迟到或取消而未用于确认的观测。stale_reads 统计通过时间戳格式检查但不满足新鲜度的样本，negative_reads 统计不满足符号后置条件的样本，二者可能重叠，不能相加当总失败帧数。duplicate_reads 只统计多样本模式中同时间戳的正面观测。streak_resets 是已有正面段被反例或陈旧样本清空的次数，gap_resets 是新正面样本因超长间隔触发重新起段的次数。

elapsed_s 从确认循环起点到最近一次时钟读取，包含轮询睡眠和 observe 返回耗时，不含 execute；它不是独立性能测量。报告不逐帧累积，默认计划最多八步，每步一个汇总。策略动作被拒绝、执行 stop 或执行前取消没有对应确认报告，之前已完成的报告保留。异常仍向上传递，没有保证在异常或进程崩溃时生成报告。

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent '把桌上的红色积木放进盒子' --json
```

JSON 写 stdout，日志按原 logging 行为写 stderr；旧文本输出保持默认。序列化结果新增字段，严格限制原 JSON 字段的外部消费者需更新。CLI 退出码行为未改，不应仅凭进程退出码判断任务成功，应读取 success。

新增三个测试覆盖报告按运行隔离、重复帧和反例计数及最终读取上限状态、独立 CLI JSON 解析与四步成功结果。与既有主运行回归共 35 项测试通过，Ruff 检查通过。

```sh
make runtime-confirmation-test
```

报告属于内存诊断，不持久化或签名，不是独立传感器证据，不授予动作权限，也没有统计成功率或实机性能提升结论。调用方可以保存 JSON 用于复盘，但报告本身不能作为重放命令。
