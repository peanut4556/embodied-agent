# 独立监督接入双视角采集后端

2026-11-16。本轮将上一轮独立作业原型接入 ProcessPairCapture，实际传输请求与 CapturedPair 帧结果，并通过 CaptureRecovery 接回预约式两阶段确认。

## 接入与资源合同

```python
backend = ProcessPairCapture(reader_factory, supervised=True)
manager = CaptureRecovery(
    position_error_m=(0.001,)*3, capture_profile='reserved', supervised=True,
)
```

默认 supervised=False，原行为保持不变。开启时每个原采集槽位由一个监督进程拥有一个实际读取进程；factory 在读取进程中创建。帧请求与结果仍走原管道，监督进程负责所有权及生命周期，不逐帧代理内容。两槽位因此最多使用两个监督进程加两个读取进程。

主进程 cancel 关闭监督存活连接，请求监督进程 terminate/kill 并 join 其读取子进程。父级 reap 保留原有时间预算，但不会因为预算耗尽而先杀监督进程；若仍未完成，返回清理未确认，可显式再次 reap。没有假定关闭请求等于清理完成。

supervised 模式下 workers 中 pid/exitcode 对应监督进程；新增 supervision 独立报告实际 worker_pid、worker_exitcode 和 kill_sent。workers_stopped 只有在监督进程结束且最终报告确认读取子进程 stopped、有退出码时才为真。监督进程异常死亡或缺失最终报告不能被当作成功回收。硬件清理确认始终为 false。

## 验证

```sh
make supervised-capture-test
make supervised-capture-check SUPERVISED_CAPTURE_OUTPUT=outputs/evaluations/supervised-capture-new
```

两次真实时钟、真实 spawn 进程运行均完成 36 个请求（31 候选 + 5 新鲜见证），最终为 confirmed_recent_evidence，关闭成功，监督和读取两级进程均确认退出。[实验摘要](evaluations/supervised-capture-v1.json)保存源代码哈希及完整状态报告位置、SHA-256。

新增 3 项测试覆盖帧传输/会话保持、两级回收、暂停读取进程后的 SIGKILL 及参数校验。原进程后端 9 项、恢复管理器 9 项、独立监督 3 项、内部 watchdog 4 项回归通过，共 28 项相关测试；Ruff 检查通过。

## 进展与边界

这使独立监督从“可终止阻塞作业”的原型推进到完整采集确认链的可选后端。此轮测试直接暂停实际读取进程并验证监督回收；调用方突然死亡的独立监督实验仍见上一轮文档，不声称本轮重跑所有崩溃组合。

读取器仍是合成双视角数据，没有真实相机或机器人。两次通过不是长期可靠性或实时性能保证；新增进程带来启动与内存成本，尚无吞吐基准。监督进程自身故障、OS 不可中断等待、设备资源释放仍需要更上层监督及硬件恢复策略。
