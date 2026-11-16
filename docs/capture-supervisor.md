# 不依赖工作进程线程的独立监督原型

2026-11-16。本轮新增 SupervisedCaptureJob 原型，将执行阻塞采集作业的进程交给独立监督进程持有。它针对上一轮工作进程内 watchdog 的边界：整个工作进程停止运行时，内部线程也无法响应调用方死亡。

## 所有权结构

调用方持有存活连接写端；监督进程持有读端并创建、拥有工作进程。监督进程独立轮询存活连接和子进程状态。调用方死亡触发 EOF，监督进程先 terminate，等待 50 ms；若子进程仍存活则 kill，并 join 核对退出，再结束自身。

```python
job = SupervisedCaptureJob(spawn_picklable_capture_job)
started = job.wait_started()
result = job.close()  # 关闭存活连接，请求监督进程清理；返回监督和子进程状态
```

close 超时不会先杀监督进程，以免破坏其子进程所有权；调用方必须查看 supervisor_alive 和事件结果，不能把 close 调用本身当作回收成功。状态始终保留 hardware_cleanup_confirmed=false。

**这是独立作业原型，不是 ProcessPairCapture 的替换后端。** 尚未接入帧传输、并行请求、预约采集或完整确认链；原有后端及 owner_watchdog 默认行为未改变。

## 验证

```sh
make capture-supervisor-test
make capture-supervisor-check CAPTURE_SUPERVISOR_OUTPUT=outputs/evaluations/capture-supervisor-new
```

真实进程实验让工作进程忽略 SIGTERM，然后自行 SIGSTOP。调用方通过 ps 确认 T 状态后直接 os._exit(93)，不执行 close。外层实验观察工作进程和监督进程，确认它们在兜底清理前都已消失。

最终两次实验全部通过，2 个监督进程和 2 个工作进程均消失。[最终记录](evaluations/capture-supervisor-v3.json)包含调用方退出码、PID、观察状态和源码哈希。v1、v2 的两次运行也均通过，保留原始目录；整理脚本 lint 条件表达式后执行 v3，与最终代码哈希一致，未将多轮结果合并成成功率。

3 项新增测试覆盖暂停工作进程的 kill/reap、正常完成和非法作业；4 项原内部 watchdog 测试回归通过，共 7 项。单测初版过早暂停尚未完成初始化的进程，观察到 SIGTERM 退出而非预期 SIGKILL；随后明确初始化等待和忽略 SIGTERM 条件，最终通过。正式实验另使用 OS T 状态确认暂停，避免仅凭等待时间声称已暂停。

## 能力边界

进展是退出保护可以在工作进程完全不执行 Python 线程时继续运行。SIGSTOP 是明确的失去执行能力模拟，不是实际相机驱动的原生 GIL 死锁实验；不能据此宣称解决所有驱动故障。

监督进程自身仍可能故障、被暂停或被同时杀死，OS 不可中断等待也可能阻碍回收。未提供监督进程的更上层高可用机制，也没有硬件安全复位、自动重启或实机验证。原型每个作业增加一个监督进程及启动成本，尚未做吞吐/延迟基准。它不涉及任何机器人动作授权。
