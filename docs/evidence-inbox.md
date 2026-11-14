# 证据队列消费边界：排队后再检查，异常后不重放

2026-11-14。接续[一次性近期证据交付](recovery-handoff.md)，增加 `EvidenceInbox` 并接入 `CaptureRecovery.queue_confirmation`。它解决一个领取后的具体缺口：证据在领取时足够新鲜，但经过队列等待或会话切换后，不应再交给下游作为近期证据。

## 协议

队列只有一个待派发槽位。调用 `manager.queue_confirmation(token, inbox)` 时，队列直接从当前交付对象消费一次，内部持有证据和发行方引用；返回给调用方的只是排队状态与跟踪 ticket，不返回证据字典。队列满或正在调用回调时，拒绝新的入队，并且不消耗下一份证据。任意已有 receipt 字典都不能作为入队来源。

`inbox.dispatch(consumer)` 在调用下游之前重新检查：

1. 来源会话是否已撤销。管理器关闭或进入回收时，即使证据之前已领取，也会设置独立的撤销标志。
2. 当前主机单调时钟是否有效、是否回退。
3. 当前时间是否仍严格早于原 receipt 的到期点。排队不会延长有效期，到期边界本身即拒绝。

期限仍来自最后实际见证曝光，不从入队或出队时重新起算。拒绝的项会被丢弃，不能再次派发。时钟异常还会锁定队列拒绝新入队。

调用下游之前先移除待派发项并设置忙状态，避免回调重入重复消费。若回调抛出普通异常，记录 `outcome_unknown/consumer_raised`，不会重试：回调可能已经产生了部分副作用，异常不能证明“什么都没发生”。回调正常返回记录 `delivered/consumer_returned`，这也只表示回调返回，不代表某个机器人任务成功。

关闭队列会丢弃尚未派发的证据，不会中断已开始执行的回调。审计只保留最近 16 次完成事件及总完成数，快照不暴露待派发证据正文。消费后仍保存历史确认，历史记录和会话撤销可以同时成立。

## 真实进程实验

合成固定位置读取器、真实预约采集进程、真实单调时钟。四种模式各运行两次；每次先完成 31 次候选采集与 5 次新鲜见证，再入队。

| 模式 | 完整确认并入队 | 实际下游调用数 | 派发结果 |
| --- | ---: | ---: | --- |
| 立即派发 | 2/2 | 每次 1 次 | `consumer_returned` |
| 排队等待 30 ms | 2/2 | 每次 0 次 | `expired_in_queue` |
| 入队后切换新会话 | 2/2 | 每次 0 次 | `session_revoked` |
| 回调记录一次后抛错 | 2/2 | 每次 1 次 | `consumer_raised`，结果未知 |

所有模式再次派发都返回 `no_pending_evidence`。切换会话的运行在新会话启动后显式停止新任务，未使用新任务重试旧消费。所有相关工作进程最终均核实退出。

这是 8 次合成确认与下游记录回调的联调实验，不是长期成功率或真实设备验证；没有调用机器人动作。旧有采集时序抖动及窄脉冲盲区仍然存在。

## 使用与边界

```python
inbox = EvidenceInbox()
manager = CaptureRecovery(position_error_m=validated_error_budget, capture_profile="reserved")
try:
    token = manager.start(raw_camera_factory)
    if token is not None:
        while manager.step(token) == "pending":
            schedule_next_poll()
        offered = manager.queue_confirmation(token, inbox)
        # 由同一控制线程在需要交付时调用；延迟、关闭或恢复可能使本次派发被拒绝。
        dispatch = inbox.dispatch(record_evidence)
finally:
    inbox.close()
    manager.close()
```

直接 `consume_confirmation` 与排队消费共享同一次领取机会，应按用途选择一条路径。先直接领取再尝试入队会被拒绝。管理器必须保留到派发时；提前关闭会撤销会话。

这是同一进程、同一所有者线程、同一主机时钟域内的应用协议。不是网络协议、签名凭据、分布式 exactly-once 保证或防篡改边界。它保证通过本 API 在回调入口检查证据，并限制每份已入队证据最多调用一次回调；不能保证回调内部不会复制数据、重复产生副作用或执行到证据过期后。消费者仍需负责自己的操作期限与业务幂等性。所有结果继续标记 `robot_action_authorized=false`，默认机器人执行路径没有启用此队列。

## 复现与验证

```sh
make evidence-inbox-test
make evidence-inbox-check EVIDENCE_INBOX_OUTPUT=outputs/evaluations/evidence-inbox-new
```

[紧凑实验记录](evaluations/evidence-inbox-v1.json)保留每次确认、入队、派发、重复调用与进程退出结果，以及源码哈希、原始报告路径和 SHA256。

新增 10 项队列测试覆盖派发一次、到期边界、撤销已领取证据、满队列反压、回调异常、重入、时钟回退锁定、关闭、拒绝字典导入和有界审计。连同采集/恢复/交付/挑战控制器回归，共 **145 项相关测试通过**；5 个涉及的 Python 文件 Ruff 及格式检查通过。
