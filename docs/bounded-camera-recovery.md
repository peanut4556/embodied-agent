# 有限次数的相机故障恢复

2026-10-29。新增实验性 `RecoveryExposureObserver`，在已有进程观测控制层上增加恢复准入与审计。默认路径未启用，无自动设备校准或机械臂动作。

## 策略

生命周期管理侧显式调用 `restart(factory=None)`，控制器从调用方提供的实时 `task_clock` 读取任务时间。它与 step 的任务时间必须同域、持续递增，不能在设备重连后归零。

只有 failed 或 timed_out 会话允许恢复。健康、启动中、待完成请求以及单帧 error 不触发重启。cleanup_failed 直接结束本次确认，防止继续产生进程。默认最多 1 次尝试，可配置非负整数；0 禁用恢复。计数在实际尝试前增加，回收或启动失败也占用次数。终态无法重启。

原始释放期限仍为 3 秒。默认剩余时间必须大于 1.25 秒才准入，至少预留 1 秒稳定证据所需时间；这只是配置门槛，不保证启动、校准和采集能及时完成。回收和 spawn 返回后再次读取任务时钟；若已到期，或时钟无效/回退，则结束确认并禁止新映射。系统调度或驱动清理可能跨越期限，此策略不能中断已经进行中的生命周期调用，也不是硬实时保证。

重启完成只表示新进程已启动。随后仍须等待 idle、提供新映射，并取得被接受的新曝光观测，才将该次尝试记为 observation_restored；它与 confirmed_in_tray 是不同结果，后者仍要求新的连续稳定证据。不会继承重启前的证据。

## 调用和诊断

```python
observer = RecoveryExposureObserver(
    session, release_task_time,
    task_clock=current_task_time,
    max_restarts=1, min_remaining=1.25,
)
try:
    # 沿用 ProcessExposureObserver 的 step 和 bind_mapping 流程。
    # 观测到失败后，在生命周期管理侧显式调度：
    spawned = observer.restart(replacement_camera_factory)
    # spawned 为 True 时，等待 idle 并重新测量、绑定当前会话映射。
finally:
    observer.close()
```

该入口仍要求单一拥有者。restart/close 可能等待进程回收，不得放入高频伺服循环。停止、超期或拒绝恢复后，调用方仍须执行 close，尤其是在新进程已启动但返回时已超期的情况下。直接调用底层 session.restart 或父类方法绕过本控制层，不受其次数限制。

result schema_version 为 2，保留状态、原因、会话和原始期限，并新增 max_restarts、restart_count、recovery_attempts。每次实际尝试记录旧会话、故障类型/错误、启动结果、新会话及完成时间（可用时）；返回记录为副本。被拒绝的恢复不会增加次数，原因记入最终 reason 和事件 rows。spawn 失败、再次故障、等待映射时停止/超期与取得有效新观测都有独立结果。已恢复观测后再发生的任务终态由顶层 status/reason 表达。

## 验证与限制

复现：`make camera-recovery-test`。

新增 10 项测试覆盖：重新映射并积累成功、重复故障达到上限、健康状态不消耗次数、预算不足与精确到期、回收/启动失败计数、清理失败拒绝、重启耗时越界及任务时钟回退、停止/映射等待超时、非法配置及禁用恢复、真实 spawn 子进程退出码 7 后替换相机并取得新观测。

其中实际进程测试验证崩溃、回收、重启、映射及新观测路径；稳定性与时限边界使用确定性会话桩。未连接真实相机，也未开展新的物理抓放或统计延迟实验。

本轮相关回归共 163 项通过：58 核心、31 接触、22 释放、13 线程采集、14 曝光、7 采集进程、8 恢复控制、10 有限恢复；Ruff 和差异检查通过。
