# 调用方崩溃后的证据发现

2026-11-14。上一轮账本按 receipt_id 对账，仍依赖调用方记住编号。本轮增加可选的 `IndexedEvidenceLedger`，在证据事务之前先提交持久化索引。调用方丢失全部内存后，可以仅凭已知数据库路径分页发现编号并核对记录。

## 提交与恢复合同

1. 校验原始 receipt，计算规范 JSON 的 SHA-256，索引绑定编号、会话和摘要；索引不保存证据正文。
2. SQLite FULL 同步模式下单独提交索引事务，然后调用原有账本事务。两次提交之间允许有缺口，明确保留该缺口供恢复核对。
3. 账本在索引落盘之后再次检查原始到期时间，索引写盘不会延长证据期限。已有完全相同记录的重复请求只能返回历史确认。
4. `discover_evidence(path, after=0, limit=100)` 只读分页（每页最多 1000 条），无需传入编号。每页使用同一读事务核对索引和账本，返回 `next_after`；并发新增条目可在后续扫描出现。空页不代表永远没有新记录。
5. 返回 `recorded`、`not_found`、`intent_conflict` 或 `integrity_error`。数据库缺失、不可读或无索引表返回 `unavailable`，不偷偷创建文件。历史普通账本里的无索引行不会自动补入索引。

接入方式（同一所有者线程，数据库应在领取证据前初始化）：

```python
with IndexedEvidenceLedger(database_path) as ledger:
    event = inbox.dispatch(ledger.record)
# 程序重启后，只需要持久保存的数据库路径：
page = discover_evidence(database_path)
```

如果崩溃留下 SQLite 热日志，先显式用可写账本连接完成 SQLite 自身回滚，再进行只读扫描。这不是重新执行 receipt。读取 API 本身不做可写恢复。

## 可复现实验

```sh
make indexed-evidence-test
make indexed-evidence-check INDEXED_EVIDENCE_OUTPUT=outputs/evaluations/indexed-evidence-ledger-new
```

本轮 10/10 案例符合预期，每种两次。独立 spawn 进程内部创建随机编号和合成审计 receipt；父进程从未收到编号或 receipt。子进程使用真实 monotonic 时钟、20 ms 到期时间、真实 SQLite 与 `os._exit`。这轮隔离验证持久化恢复，没有重新运行完整相机确认链。

| 故障点 | 退出码 | 重启后发现结果（每种 2 次） |
| --- | --- | --- |
| 正常提交 | 0 | 已落账 |
| 索引 COMMIT 前退出 | 71 | 无条目 |
| 索引提交后、证据事务开始前退出 | 73 | 有索引，未落账 |
| 证据 COMMIT 前退出 | 72 | 有索引，未落账 |
| 证据提交后退出 | 74 | 已落账 |

[提交的实验摘要](evaluations/indexed-evidence-ledger-v1.json)包含源文件哈希、原始报告位置/哈希、各数据库哈希和完整核对状态。36 项证据链测试通过，其中新增 10 项覆盖无编号发现、分页、过期、摘要绑定、损坏、锁竞争、缺失数据库和实际队列回调接入。Ruff 检查通过。

## 能力边界

突破点是恢复线索从调用方内存迁移到持久化索引，使重启后能够主动发现并解释中断的提交。索引提交前崩溃仍无法恢复该次编号；`not_found` 只表示没有本地证据记录，不能证明外部动作未发生。扫描不会自动重放、刷新时效或返回动作授权。

两次事务增加同步写入开销，可能消耗掉严格的 20 ms 时效；超时应拒绝，未承诺实时性能。测试覆盖进程退出，不覆盖断电或磁盘损坏耐受性。摘要用于一致性核对而非防篡改认证。单页有界，累计索引尚无清理策略；扫描定位依赖数据库路径仍然可获得。这是可选本地审计适配器，未接入真实机器人、头显或生产 UI，也不构成分布式 exactly-once。
