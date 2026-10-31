# 释放确认后的独立视觉监测

2026-10-31。新增实验性 PostConfirmationWatch，为已经 confirmed_in_tray 的释放结果提供独立、有限时长的后续观测窗口。原确认结果保持不变，后续观测状态和区域外告警单独记录。

## 监测语义

构造时显式提供成功确认结果、监测起点、相机会话与 ExposureClockMap。默认监测 2 秒。调用方必须保证起点是实际成功之后的任务时间，映射与会话有效；模块不会自行启动相机、重启进程或发送机械命令，也不会重新校准时钟。

只有晚于监测起点的新曝光才可用于后续判断，原成功帧不能重复利用。复用现有双时钟映射的年龄、漂移、未来时间检查，同时拒绝重复曝光、无效坐标和任务时间回退。

观测状态分为：

| evidence_state | 含义 |
| --- | --- |
| unknown | 尚无后续证据、观测缺失/被拒绝，或窗口已停止/结束 |
| inside_region | 最近一次有效观测在原保守确认区域内 |
| outside_candidate | 有区域外观测，但连续时间不足 |
| outside_region | 区域外曝光连续覆盖至少 250 ms，且相邻间隔不超过 250 ms |

缺失、无效、区域内观测或超长曝光间隔都会打断区域外累计。告警依据的是多个采样点，不证明采样之间连续处于区域外；250 ms 是本实验策略参数，尚未经真实相机标定。该区域比物理托盘保守，因此事件命名为 outside_confirmation_region_detected，不能将其解释为确定掉落或掉出托盘。

首次达到区域外条件会留下一个历史告警。返回区域内会更新当前 evidence_state，但不会清掉 outside_detected，也不会重复记录相同监测窗口内的后续告警。原成功结果以独立深拷贝保留。

状态 status 从 monitoring 进入 completed（到期）、cancelled（停止）或 invalidated（会话变化）后锁定。停止优先于时间校验；到期不消费传入帧，并将当前观测状态置为 unknown，避免最后一次区域内观测被解释为持续有效。

## 调用边界

```python
watch = PostConfirmationWatch(
    confirmed_result, confirmation_delivery_time, camera_session_id, measured_clock_map
)
watch.step(task_now, host_now, stamped_position_or_none, session_id=camera_session_id)
report = watch.result()
```

这是接收观测数据的单一拥有者状态机，尚未接入 ProcessExposureObserver 或 RecoveryExposureObserver。接入方需要显式调度后续采集，并在观测缺失时传入 None、在停止/到期时调用 step；若完全停止调用，结果只代表上次调用时的快照，不具备后台失效计时。采集进程的关闭/回收仍归调用方管理。

不把完成监测或“没有告警”视为物体一直安全；缺失数据可能覆盖整个窗口。持续物理安全监控和自动补救动作均不在本模块范围内。

## 验证

[8 类场景报告](evaluations/post-confirmation-watch-v1.json) 先通过现有 ExposureConfirmation 得到成功结果，再分别测试区域内、瞬时区域外、持续区域外、缺失、旧时间戳、会话变化、停止、告警后返回。场景保留输入帧、逐步快照、到期前与到期后的状态。

```sh
make post-confirmation-test
make post-confirmation-check
```

默认输出 outputs/post-confirmation-watch-v2.json，可用 POST_CONFIRMATION_OUTPUT 指定新路径，已有文件不覆盖。

新增 9 项测试覆盖持续告警、短暂跳点、中断与间隔、过期/未来/旧成功帧拒绝、会话及停止优先级、到期不复用、无效输入/漂移、结果隔离和原确认到后续监测的场景衔接。相关回归共 213 项通过（58 核心、31 接触、34 释放、13 线程采集、14 曝光、7 采集进程、8 进程恢复、39 恢复相关、9 后续监测），Ruff 与差异检查通过。

本轮使用合成带时间戳坐标，未连接真实相机，也未执行新的物理抓放或训练。下一步可将该模块显式接入进程采集生命周期，并验证监测到期和异常时的资源回收。
