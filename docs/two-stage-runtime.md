# 两阶段确认的真实线程运行适配器

2026-11-12。本轮将[统一采集状态机](two-stage-capture.md)接到真实后台工作线程，并使用真实单调时钟做运行与退出检查。相机数据仍由合成读取函数产生，没有接通真实设备或共享 MuJoCo 渲染器。

## 新增运行组件

`ThreadedPairCapture` 接受 `read(request, cancel_event)`，返回包含 session 和两份 `StampedPosition` 的 `CapturedPair`。最多两次调用/未消费结果占用槽位，没有隐藏的无限执行队列；每次运行最多 36 个不同请求，重复编号拒绝。控制器阶段内的更严格在途限制继续有效。

读取函数在实际后台线程执行，结果按发布完成的顺序交付，不通过请求编号排序掩盖乱序。驱动异常转为最多 512 字符的错误记录，不能产生成功观测。`cancel` 关闭入口、设置取消事件、丢弃已排队或后来返回的结果。控制线程负责 submit/drain/cancel/reap，读函数必须独占或正确同步相机资源；不得共享正在修改的仿真状态或非线程安全渲染器。

`TwoStageCaptureRuntime` 连接这些工作线程与两阶段控制器：

- `step()` 先消费已完成结果，再发下一请求，以控制器真正消费时刻计算交付时间，排队时间不会被隐藏。
- 返回值只有状态字符串；完整历史用 `result()` 显式获取，不在高频轮询中复制。
- 候选依旧只作为中间证据，最终成功仍由顶层两阶段控制器判断。
- 驱动失败、容量不可用、时钟异常或外部停止都会关闭采集入口；异常传播前先发出取消信号。
- 停止不读取外部时钟，因此时钟损坏也不会阻止停止请求。
- `close(wait_s=...)` 设置终态、取消读取，并在总计最多 250 ms 的显式等待预算内检查线程退出。调用方应在 `finally` 中调用。

结果明确区分 `closed`（不再接收新工作）、`workers_stopped`（线程是否已退出）、`live_request_ids`（仍存活请求）和 `hardware_cleanup_confirmed=false`。线程退出不等于相机/硬件已经完成清理，驱动必须自行管理设备资源。

Python 线程不能强制终止卡住的原生驱动调用。本实现采用有界 daemon 工作线程：卡住后不再启动新任务，但该线程仍占资源。需要强制终止的驱动应放入独立进程；本轮没有用线程取消标记冒充强制退出。

## 高频路径修复

初版 `step()` 每次返回完整的深复制快照。真实时钟试运行中，快速模拟驱动在第 11 个请求后错过计划时刻，结果为 `capture_schedule_missed`；该次运行观测到的最大单步耗时约 6.98 ms，超过 2 ms 请求迟到窗口。

将完整快照移出高频路径后重新运行，快速路径完成全部 36 个请求。确定性测试还验证 `step()` 不会构造完整审计快照。两次实际运行并非受控性能基准，不能把全部差异归因于一个因素，也不能据此推断操作系统实时性保证。修复后那次运行的最大单步仍约 3.21 ms，只是本次没有因此错过必须满足的采集时刻。

## 真实线程与真实时钟检查

脚本运行三个合成读取模式。快速模式以线程内当前主机单调时间作为曝光时间，候选阶段等待约 2 ms，见证阶段约 1 ms；这些等待是测试注入值，不是真实图像处理耗时。

| 模式 | 最新运行结果 | 取消后的线程状态 |
| --- | --- | --- |
| 快速合成读取 | 36 请求，`confirmed_recent_evidence`；运行约 3.202 秒 | 全部退出 |
| 合作式卡住 | 2 请求后 `capture_request_timeout` | 读函数响应取消，全部退出；迟到结果丢弃 |
| 不合作式卡住 | 2 请求后 `capture_request_timeout` | 20 ms 回收等待后仍有 2 线程存活；测试解除阻塞后才全部退出 |

不合作读取通过测试事件模拟，最终由测试显式释放，不是运行器杀死了线程。两条卡住路径始终未将超时当作成功，不会启动无限重试。

另外，使用真实工作线程配合可控虚拟时钟完成两阶段全链测试，验证全部 36 个请求和最终确认；该测试用于确定性回归，不用于测量墙钟吞吐。

## 接入方式

```python
controller = TwoStageCaptureController(
    0, ExposureClockMap(host_origin, 0), camera_session,
    position_error_m=validated_error_budget,
)
runtime = TwoStageCaptureRuntime(controller, read_pair)
try:
    while runtime.step() == "pending":
        # 由调用方调度下一次轮询；必须满足所选协议的时间窗口。
        schedule_next_poll()
    final_snapshot = runtime.result()
finally:
    cleanup = runtime.close(wait_s=0.02)
    # workers_stopped=False 时仍需跟踪驱动，不能声称资源已全部释放。
```

`read_pair` 必须返回真实曝光时刻和正确 session，且返回值来自请求之后的新采集。实际接入还需要满足位置误差预算、双视角同步、取消约定和足够频繁的调用方轮询。运行器没有自动创建控制循环、网络连接或执行机器人动作。

## 复现与验证

```sh
make two-stage-runtime-test
make two-stage-runtime-check TWO_STAGE_RUNTIME_OUTPUT=outputs/evaluations/two-stage-runtime-new
```

[初版记录](evaluations/two-stage-runtime-v1.json)与[最终记录](evaluations/two-stage-runtime-v2.json)都保留源文件哈希、真实时长、状态、回收前后线程列表，以及完整报告路径/SHA256。原始报告保留请求和证据链。输出目录独占创建，复跑不得覆盖旧记录。

新增 10 项测试覆盖容量与无隐藏队列、完成顺序、驱动异常及单次消费、真实线程两阶段链路、坏时钟下停止、运行异常时取消、错误返回类型、参数校验、不可合作读取和审计快照移出热路径。连同原相关回归共 **184 项通过**，Ruff 通过。

当前仍未接通真实相机，未证明设备误差和延迟预算同时达标，也没有解决窄脉冲盲区。默认执行端不切换到本适配器。下一步应选择一个真正可取消、资源独立的观测驱动，实测曝光到交付延迟和位置误差，再决定是否适合当前严格时间窗口。
