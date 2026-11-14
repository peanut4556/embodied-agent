# 恢复后的近期证据交付：一次领取、过期拒绝

2026-11-14。把[预约式两阶段确认](reserved-two-stage.md)接入[有界采集恢复](capture-recovery.md)，并新增 `RecentEvidenceHandoff`。现在可以明确区分“历史上确认通过”和“此刻仍可领取近期证据”。本模块不授予机器人动作权限，也不替代原有监测或动作安全检查。

## 恢复与预约接通

`CaptureRecovery(..., capture_profile="reserved")` 会在每次显式 start 时，用 `ScheduledReaderFactory` 包装原始相机工厂，预热后建立新的 `ReservedTwoStageController` 和交付对象。默认配置仍是 `immediate`。

旧进程全部退出并核实退出码之后，才能创建新会话；新会话重建候选、见证、请求列表和领取状态，不继承旧成功标志。旧令牌不能驱动或领取新会话。启动失败仍消耗次数，最多三轮，不自动重放任务。

进入关闭或回收时，未领取的交付立即撤销，即使进程回收尚未完成，也不能继续领取旧会话证据。历史报告保留该轮交付状态；它不会随新会话启动而改成新结果。

## 一次性交付合同

`consume_confirmation(session_id)` 只允许完整的 `confirmed_recent_evidence` 结果进入时效检查。候选完成、任务未完成、错误令牌、已撤销、已领取或已过期的交付都不会返回证据。

成功领取前先构造证据快照，再读取主机单调时钟，因此复制快照的耗时也计入时效预算。到期点固定为：

```text
最后一次实际见证曝光时刻 + 20 ms（任务时钟预算换算到主机时钟）
```

不是“确认判定时刻 + 20 ms”，也不是“请求领取时刻 + 20 ms”。到达截止点即拒绝。时钟异常或回退会锁定拒绝，过期也不能通过重新轮询或恢复时钟值续期。

一个控制器实例只能绑定一个交付对象；领取成功后再次调用返回 `already_consumed`。历史确认记录仍保留，不被伪装成从未发生过的失败。返回值包含最后曝光时间、领取时间、绝对到期时间和当时剩余时间，始终标记 `robot_action_authorized=false`。

这里的“一次性”指同一进程、同一控制器的 API 交付语义，不是防篡改或跨进程防重放机制。调用方仍能复制已经收到的字典；系统无法靠这个对象撤回已被外部保存的数据。`remaining_s` 是领取瞬间的记录，不是实时倒计时。传输或后续处理也会消耗剩余时间，接收方必须在同一有效时钟域中重新比较截止点。任何机械动作仍需独立授权和执行时的适用性检查。

## 真实进程实验

运行四组，每组两代会话：旧读取器忽略 SIGTERM 并卡住，直到请求超时；调用方再显式启动快速读取器。新会话使用预约式完整确认。位置和观测均为合成数据，使用真实进程与真实单调时钟。

| 领取条件 | 新会话完整确认 | 领取结果 | 再次领取 |
| --- | ---: | --- | --- |
| 立即领取，第 1 次 | 36 请求，通过 | 成功，剩余约 11.11 ms | `already_consumed` |
| 等待 30 ms，第 1 次 | 36 请求，通过 | `expired` | `expired` |
| 立即领取，第 2 次 | 36 请求，通过 | 成功，剩余约 11.71 ms | `already_consumed` |
| 等待 30 ms，第 2 次 | 36 请求，通过 | `expired` | `expired` |

四组旧会话均为 `capture_request_timeout`；旧令牌在新会话中均为 `session_mismatch`。所有八代会话的工作进程最终均核实退出。延迟领取的两组保留了成功的历史确认，但没有返回可领取的近期证据。

这四组展示恢复、确认与领取的衔接，不是恢复可靠率或真实设备性能统计。此前预约模式发生过的时序失败仍成立，本轮没有解决调度抖动、窄脉冲漏检、设备重开条件或硬件安全。

## 调用方式

```python
manager = CaptureRecovery(position_error_m=validated_error_budget, capture_profile="reserved")
try:
    token = manager.start(raw_camera_factory)
    if token is not None:
        while manager.step(token) == "pending":
            schedule_next_poll()
        receipt = manager.consume_confirmation(token)  # 在关闭之前，显式领取一次。
        # 检查 receipt.status 和截止点；不能直接当成机械动作许可。
finally:
    fully_reaped = manager.close()
```

同一管理器由一个控制线程调用。若要恢复，需要显式再 start，并使用新返回的 token；设备是否可重新打开仍由驱动负责验证。默认控制台与机器人动作路径没有自动启用本功能。

## 复现与验证

```sh
make recovery-handoff-test
make recovery-handoff-check RECOVERY_HANDOFF_OUTPUT=outputs/evaluations/recovery-handoff-new
```

[紧凑实验记录](evaluations/recovery-handoff-v1.json)包含每组确认和领取结果、源码哈希、完整报告路径与 SHA256。输出目录独占创建。

新增 8 项交付测试和 2 项恢复集成测试，覆盖重复领取、错误会话、曝光起算的过期、复制快照耗时、时钟异常锁定、撤销、预约恢复和清理失败。连同采集及挑战控制器回归，共 **135 项相关测试通过**；5 个涉及的 Python 文件 Ruff 及格式检查通过。
