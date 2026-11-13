# 可终止的双视角采集进程后端

2026-11-13。本轮给[两阶段运行器](two-stage-runtime.md)增加 `ProcessPairCapture` 后端，解决线程后端无法结束卡住原生读取的问题。仍使用合成读取函数，没有接通真实相机，默认后端保持线程模式。

## 隔离与生命周期

每个运行器最多预热两个持久 `spawn` 子进程。工厂在各自子进程里创建独立读取对象，进程之间不共享相机句柄或仿真渲染状态。工厂必须可被 spawn 序列化；不能把父进程已打开的设备对象直接传入。

`wait_ready()` 在任务时钟启动前完成初始化，最多等待五秒。初始化失败、子进程退出或启动超时会关闭入口，调用方仍需在 finally 中执行 cancel/reap。运行器只接受已预热的就绪后端，不把进程创建开销塞进 2 ms 请求时间窗口。

每个子进程最多一个在途请求，无隐藏任务队列，运行器总请求上限仍为 36。返回结果携带请求编号、session、真实曝光时间与有限的三维坐标；子进程在发送之前规范化坐标、限制 session 和错误字符串长度。独立管道属于本次后端实例，不自动重启或复用旧会话。

若多个结果已经就绪，按子进程记录的单调完成时刻合并，不按请求编号排序来掩盖乱序；原控制器仍会拒绝已知请求的乱序交付。工厂或读取异常、EOF、子进程意外退出均不会产生成功观测。

生命周期接口与线程后端一致：submit、drain、cancel、reap、result。`TwoStageCaptureRuntime(..., capture_backend=backend)` 使用该后端；不能同时提供线程 read 函数。

## 真正终止与诚实报告

- `cancel()` 关闭提交入口和父端 IPC，并对仍存活进程发送终止信号，不在高频 step 中等待退出。
- `reap(wait_s)` 的等待预算最多 250 ms：前半段等待终止，仍存活时升级 `kill`，剩余预算等待退出。
- 已确认退出的进程关闭管理句柄，保存 PID、退出码和是否使用 kill。未退出的进程继续报告存活，调用方可以随后再次 reap；不会伪造成功回收。
- 关闭后不再接收或消费结果。没有自动重启或重叠重试。

这些是有界等待策略，不是操作系统调度或任意不可中断内核状态下的严格退出时限保证。强制退出可能绕过设备清理代码，因此结果始终保留 `hardware_cleanup_confirmed=false`。只有驱动/设备层另行确认，才能宣称硬件状态已恢复。相机进程退出也不代表机器人已执行安全停止。

## 真实进程、真实时钟检查

本地 POSIX 环境运行三种合成工厂，各只测一次，结果不构成设备性能统计：

| 模式 | 结果 | 退出证据 |
| --- | --- | --- |
| 快速合成读取 | 完成 36 请求，`confirmed_recent_evidence` | 两个进程退出，退出码 −15，无存活 PID |
| 忽略 SIGTERM 的永久阻塞读取 | 两请求后 `capture_request_timeout` | 两进程升级 kill，退出码 −9，无存活 PID |
| 工厂初始化抛异常 | 未启动确认任务、零采集请求 | 子进程全部回收 |

本次快速模式预热约 55 ms，关闭回收约 2.4 ms；阻塞模式回收约 130 ms。初始化失败模式约 76 ms 返回未就绪。具体时长保存在[摘要](evaluations/process-pair-v1.json)，只代表这一轮模拟读取运行。

关键进展是：上一轮不合作线程只能在测试显式释放后退出，本轮忽略普通终止信号的阻塞读取无需解除阻塞，即可由父进程强制结束并核实退出。测试函数只是永久睡眠循环，并未证明所有真实原生驱动或内核阻塞都能在相同时限内退出。

## 接入方式

```python
backend = ProcessPairCapture(camera_factory)
try:
    if not backend.wait_ready():
        raise RuntimeError("camera workers not ready")
    origin = time.monotonic()  # 预热之后启动任务时钟
    controller = TwoStageCaptureController(
        0, ExposureClockMap(origin, 0), camera_session,
        position_error_m=validated_error_budget,
    )
    runtime = TwoStageCaptureRuntime(controller, capture_backend=backend)
    while runtime.step() == "pending":
        schedule_next_poll()
    result = runtime.result()
finally:
    backend.cancel()
    cleanup = backend.reap(0.25)
```

子进程读取对象接口为 `reader(request) -> CapturedPair`；与线程版 `read(request, cancel_event)` 不同，进程版通过终止进程实现取消。工厂应创建独立资源；同一个物理相机是否允许多个读取进程，需要驱动明确支持，不能因为默认有两个槽就重复打开不支持并发的设备。也可以显式设置容量一，吞吐不足时由原控制器拒绝，不自动绕过时序约束。

## 验证与复现

```sh
make process-pair-test
make process-pair-check PROCESS_PAIR_OUTPUT=outputs/evaluations/process-pair-new
```

摘要包含源码哈希、预热与回收时长、退出码、kill 标记、最终状态及完整报告路径/SHA256。原始报告保留请求及证据快照，输出目录独占创建。

新增 8 项测试覆盖预热和单次结果、重复请求、容量限制、忽略 SIGTERM 的 kill 回收、工厂失败、运行器停止、参数校验、进程意外退出、启动卡住和完成顺序。连同原线程、两阶段及相关回归共 **192 项通过**，Ruff 通过。原线程后端的真实工作线程与取消测试保持通过。

默认执行端未切换；位置误差、真实相机延迟、窄脉冲盲区和设备恢复都没有由进程隔离自动解决。下一步可以在此后端上接入可独立打开、能够提供曝光时间戳的实际观测源，先验证设备资源恢复与时延，再决定是否满足当前确认协议。
