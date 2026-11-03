# 采集进程驱动的告警复查

2026-11-03。新增 `ProcessPostAlertVerification`，将[告警后的有限视觉复查](post-alert-verification.md)接入 `CameraProcessSession`。从成功释放确认显式交接同一相机会话，持续请求后续帧；区域外告警触发后，无需外部逐帧喂入坐标即可进入新证据复查。

## 使用方式

```python
from embodied_agent.process_post_alert_verification import ProcessPostAlertVerification

# observer 已通过 ProcessExposureObserver 或兼容恢复观察器获得成功确认。
with ProcessPostAlertVerification.from_confirmed(observer) as flow:
    observer.close()  # 已交接，旧控制器不会关闭相机。
    while flow.status in {"monitoring", "rechecking"}:
        flow.step(task_clock(), stop=stop_requested())
        # 调用方负责合适的轮询调度；step 不等待相机 read，也不 join。
report = flow.result()  # 此时包含上下文退出后的 cleanup 结果。
```

也支持显式构造，提供已有成功结果、起点、空闲/就绪相机会话和有效时钟映射。来源没有成功确认、存在未完成请求或重复交接时拒绝接管。新适配器与既有 `ProcessPostConfirmation` 共用独占会话标记，不能同时占用同一相机；原任务结果不改写。

实现复用既有适配器的交接、前后时钟检查与清理机制，覆盖两阶段调度和输出接口。阶段为 monitoring / rechecking 时允许采集，其余终态不再 poll/request。close 或上下文退出负责取消仍在运行的阶段并回收进程；step 本身不执行回收。

## 调度与失效规则

- 请求未完成的空轮询只执行时间与会话检查，不作为“缺帧”送入稳定证据计算。状态机新增 waiting 参数支持这种心跳，默认行为保持不变。
- 一次 pending → ready/error 才消费一个结果。明确缺失、拒绝或错误会打断证据累计。
- 等待期间，最近接受的曝光超过映射 max_age 后，以一次 expired 事件打断累计；后续空轮询不会重复记失效。
- 监测与复查各有原定截止时间；告警不会重置原确认结果，缺帧不会延长复查预算。适配器在截止点先结算，不再消费管道中已缓冲的帧，也不再发送新请求。这比直接帧输入状态机在截止点处理新帧的规则更保守。
- 停止先于时钟检查；非法任务时间、相机会话/拥有权改变、前后时钟漂移、请求拒绝、采集超时或崩溃均终止流程。poll 后再次核对拥有权，防止在采集期间发生更换后继续消费。
- close 只回收自己拥有的会话，不能关闭其他拥有者的替换会话。清理失败单独记录并允许重试；业务异常和清理异常同时发生时保留两个异常。

结果保留原成功、历史告警、新确认、事件及清理状态。不会把无告警、无图像或回收成功当成放置安全。此适配器没有新增覆盖率统计字段；原有覆盖率工具继续针对其原来的压力报告使用。

## 实际运行结果

[运行报告](evaluations/process-post-alert-v1.json)使用真实 spawn 子进程和 Pipe 通信，坐标由子进程内模拟相机按实际单调时钟产生。四条路径均先采集区域内观测形成初始成功，再交接，并制造持续区域外告警：

| 告警后的情况 | 结果 | 进程回收 |
| --- | --- | --- |
| 返回区域内并稳定 | reconfirmed | 成功 |
| 相机 read 卡死 | invalidated / camera_unavailable | 成功 |
| 子进程以退出码 7 崩溃 | invalidated / camera_unavailable | 成功，保留退出码 |
| 用户停止 | cancelled | 成功 |

4/4 符合预期；终态后没有新增请求，原确认结果均保持不变。正常回收使用 terminate，因此回收后的负退出码不表示业务确认失败。记录中的请求数和时间受主机调度影响，不作为固定回归断言。

```sh
make process-post-alert-test
make process-post-alert-check
```

检查默认输出 `outputs/process-post-alert-v2.json`，用 `PROCESS_POST_ALERT_OUTPUT` 指定其他未占用路径。示例需要 spawn 能力，预计运行约十秒；主机负载过高可能导致采集时限内无法完成，检查会失败而不会放宽门槛。

新增 11 项测试覆盖 pending/expiry 区别、复查截止、停止、双适配器排他、双异常及清理重试、poll 中漂移、请求失败/非法时刻、待处理会话拒绝交接，以及真实子进程成功、卡死、崩溃路径。相关回归共 109 项通过：本轮 11、纯复查 11、原进程后续监测 10、后续边界/覆盖/矩阵 48、曝光 14、采集进程 7、进程曝光观察器 8。Ruff 与差异检查通过。

这是进程采集与视觉决策的端到端流程，尚未使用真实相机、驱动物理机械补救或接入网页。模拟物体返回是测试输入，不是系统执行了抓取恢复；真实环境的可靠性仍需后续设备实验验证。
