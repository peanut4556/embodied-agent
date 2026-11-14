# 崩溃后的证据账本对账

2026-11-14。接续[证据队列消费边界](evidence-inbox.md)，新增 SQLite 证据账本与只读对账接口。目标是回答一个具体问题：队列记录回调结果未知时，本地证据事务究竟有没有提交？它不改变原有“不自动重放”的规则，也不判断机器人动作是否完成。

## 从领取编号到持久化记录

每份成功领取的证据增加 UUID `receipt_id`。同一编号贯穿入队返回值、队列审计与账本主键；入队返回值仍不暴露证据正文。调用方可以在派发前保存这个编号，后续按编号对账。

`EvidenceLedger.record(receipt)` 适合作为本地证据记录回调：

- 规范化序列化 receipt，拒绝非有限 JSON 数值和超过 1 MiB 的单条记录。
- 在一个 SQLite 事务中绑定编号、会话、完整快照及 SHA256。
- 相同编号和相同内容的重复写入只返回已存在的历史记录；相同编号但不同内容会冲突，不能覆盖。
- 序列化和取得写锁的耗时计入新写入的时效预算，插入前及事务提交前均检查主机时钟与原截止点；超限则回滚。
- 写锁等待配置为零，争用直接失败，不把过期的请求无限排队。数据库采用 rollback journal 和 `synchronous=FULL`。

重复确认已有记录不代表证据重新变新；旧记录只用于审计。账本不自动删除历史，部署时还需选择保留策略。原确认、领取与队列派发逻辑仍负责各自的证据边界。

## 对账语义

`reconcile_evidence(path, receipt_id)` 以只读方式打开已有数据库，不会在路径不存在时创建一个空账本冒充“没有记录”：

| 结果 | 含义 |
| --- | --- |
| `recorded` | 该编号存在，保存的正文与摘要一致 |
| `not_found` | 数据库和表可读，但查无此编号 |
| `unavailable` | 数据库不存在、不可读、锁冲突或 SQL 读取失败 |
| `integrity_error` | 正文与保存的 SHA256 不一致 |

对账返回元数据，不重新发放证据正文，不更改队列原有的 `outcome_unknown` 记录，不触发回调重试。`not_found` 也不证明其他外部副作用没有发生；它只说明这个本地账本里没有对应记录。

进程异常退出后 SQLite 可能需要可写打开才能恢复热日志。实验先显式用账本写入器打开并关闭数据库，让 SQLite 完成日志恢复，再执行独立只读对账；这个过程没有重新提交 receipt。

## 真实崩溃实验

预热一个独立账本进程，再用真实采集进程和合成位置观测完成预约式 36 次采集确认。将队列回调接到该账本进程，三种模式各运行两次：

| 模式 | 写入进程退出码 | 队列结果 | 对账结果 | 回执 |
| --- | ---: | --- | --- | --- |
| 正常提交 | 0 | `delivered` | 两次均 `recorded` | 每次 1 个 |
| SQL 插入后、COMMIT 执行前退出 | 71 | `outcome_unknown` | 两次均 `not_found` | 无 |
| COMMIT 完成后、回执发送前退出 | 72 | `outcome_unknown` | 两次均 `recorded` | 无 |

故障由写入进程内部 `os._exit` 注入，提交前场景用 SQLite trace callback 在 COMMIT 执行前退出。不是把一个普通异常当作进程崩溃。所有六次运行均先完成了合成证据确认，所有再次派发都被队列拒绝，没有自动重放。采集与账本进程最终均核实退出。

这验证了“提交成功但回执丢失”与“事务没有提交”在本地账本上的区别。没有测试主机断电、磁盘硬件故障、网络分区或机械执行结果；6 次运行也不是可靠率统计。

## 接入和限制

```python
with EvidenceLedger("outputs/local-evidence.sqlite") as ledger:
    offered = manager.queue_confirmation(token, inbox)
    outcome = inbox.dispatch(ledger.record)
    if offered["status"] == "queued":
        audit = reconcile_evidence("outputs/local-evidence.sqlite", offered["receipt_id"])
        # audit 只用于记录核对，不授权自动重试或机器人动作。
```

本轮提供的是本地事务性记录，不是分布式 exactly-once 保证。持久化主键不能使任意外部系统的副作用自动具备幂等性。SHA256 用于一致性检查，不是签名或身份认证；receipt 字典及数据库仍属于可信应用的数据边界。

时钟检查发生在提交前，无法保证文件系统同步完成恰好仍在截止点内；数据库保存的是历史审计，而不是机械执行许可。保存的单调时钟时间戳也不能跨机器或重启后直接解释成新的有效期。所有结果继续标记 `robot_action_authorized=false`。默认控制台和机器人执行路径未接入账本。

## 复现与验证

```sh
make evidence-ledger-test
make evidence-ledger-check EVIDENCE_LEDGER_OUTPUT=outputs/evaluations/evidence-ledger-new
```

[紧凑实验记录](evaluations/evidence-ledger-v1.json)包括六次事务/回执/对账结果、退出码、源码哈希、原始报告及 SQLite 文件路径与 SHA256。原始数据库留在独占创建的实验输出目录中。

新增 8 项账本测试覆盖重新打开后的记录查询、重复和冲突、过期回滚、事务内时间超限、数据库不可用、正文摘要不一致、格式/容量校验和时钟回退。连同采集、恢复、交付、队列及挑战控制器回归，共 **153 项相关测试通过**；5 个涉及的 Python 文件 Ruff 和格式检查通过。
