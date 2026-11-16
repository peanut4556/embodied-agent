# 调用方卡住时的独立采集运行时限

2026-11-16。本轮增加可选 runtime_limit_s。此前存活连接只能察觉调用方退出/关闭，调用方仍活着但卡住时连接不会断开；现在监督进程可独立计时，终止超过运行时限的采集作业。

```python
job = SupervisedCaptureJob(capture_job, runtime_limit_s=10)
backend = ProcessPairCapture(factory, supervised=True, runtime_limit_s=10)
manager = CaptureRecovery(
    position_error_m=(0.001,)*3, capture_profile='reserved',
    supervised=True, runtime_limit_s=10,
)
```

默认 None 保持原行为。配置接受有限秒数 (0, 60]，拒绝布尔值、NaN/无穷和越界值；后端与恢复管理器必须同时指定 supervised=True。运行时限从监督进程开始启动实际工作进程前计时，覆盖该工作进程的初始化、空闲等待和全部请求，不是逐帧超时，也不因新请求自动续期；不包含外层监督进程尚未启动的等待时间。

监督循环用 monotonic 检查截止时间，超时后沿用 terminate→等待 50 ms→必要时 kill→join 的回收路径。最终报告新增 runtime_limit_exceeded、runtime_limit_s 和 elapsed_s。父进程暂停时仍可完成这一过程；没有把检查逻辑放进调用方或读取器。

## 故障验证

```sh
make capture-runtime-limit-test
make capture-runtime-limit-check CAPTURE_RUNTIME_LIMIT_OUTPUT=outputs/evaluations/capture-runtime-limit-new
```

设置 1 秒时限；合成工作进程忽略 SIGTERM 并永久 sleep。调用方确认工作进程启动后自行 SIGSTOP。外层实验通过 ps 核对调用方确实暂停，等待监督进程按时限处理；然后才恢复调用方，让它读取最终状态和完成 join。

最终两次均通过：调用方仍处于 T 状态时读取进程已消失，监督进程已退出；调用方恢复后监督进程也被最终回收。报告均为 runtime_limit_exceeded=true、owner_connection_closed=false、读取退出码 -9，耗时约 1.055 秒。**这是 1 秒触发阈值加终止等待和实际调度的观测值，不是硬实时 1 秒内回收保证。**

首轮两个用例被实验判为失败，因为它要求暂停父进程的子监督进程也立即从进程表消失。实际读取已回收、监督已退出为 Z 状态，但父进程暂停无法 join；这一状态不代表仍在执行。保留 v1 数据，修正实验为分别核对停止执行、恢复后的最终回收，v2 两次通过。[最终报告](evaluations/capture-runtime-limit-v2.json)保留前后 OS 状态、完整监督报告和源代码哈希。

新增 3 项测试覆盖存活调用方下超时、后端自动终止及配置校验；与原监督、采集和恢复测试共 32 项相关测试通过，Ruff 检查通过。

## 边界

这是独立作业生命周期上限，不是任务成功期限或机器人安全控制。设置过短会中止合法采集，需按业务选择；本轮未默认启用，也未调整原曝光/证据时效门槛。没有自动重启或重复执行任务。

如果监督进程本身也被暂停或死亡、OS 无法终止工作进程，仍不能保证及时回收。父进程不恢复时，已退出监督进程可能保留僵尸条目，需上层进程管理处理。真实相机、硬件资源复位、断电可靠性和长期吞吐尚未验证。
