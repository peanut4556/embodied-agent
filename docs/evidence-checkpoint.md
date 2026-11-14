# 数据库身份与历史检查点

2026-11-14。本轮把证据恢复从“找到一个能打开的数据库”推进到“先核对它是否仍保留已知历史”。旧版发现接口无法区分从未落账与备份回退导致的丢失；新增可选检查点接口，在历史不一致时停止输出对账条目。

## 工作方式

`CheckpointedEvidenceLedger` 继承上一轮索引账本，初始化时持久化随机数据库身份；已有身份证明保持不变。它仍可直接作为 `inbox.dispatch(ledger.record)` 的消费者，创建连接应在领取短时证据之前完成。

`checkpoint_evidence(path)` 在一个 SQLite 读事务中生成小型 JSON 检查点，包括数据库身份，以及索引表、证据表各自的高水位、行数和 SHA-256。摘要逐行读取，涵盖完整证据正文及其元数据，而不只相信记录内保存的摘要。检查点本身不包含正文。

调用方需要把检查点独立持久保存，避免与数据库一起恢复到旧版本。本轮实验把检查点放在独立 JSON 文件，仅替换数据库；这不是已部署的远端存储或断电可靠写入服务。

`discover_checkpointed(path, checkpoint, after=0, limit=100)` 首先验证历史，再分页发现记录。校验和读取共享同一读事务：即使其他连接在两者之间提交修改，本次仍读取先前一致的快照，下次则发现修改。

| 返回状态 | 含义 |
| --- | --- |
| `scanned`，`checkpoint_verified=true` | 检查点覆盖的历史保留，可以解释当前快照中的对账结果 |
| `database_mismatch` | 数据库身份不同，返回空条目 |
| `history_mismatch` | 已知历史缺失或改变，返回空条目，并指明索引或证据部分 |
| `unavailable` | 数据库缺失、模式不兼容、无法读取等；不创建文件，也不自动登记新身份 |

正常追加不改变已核对的历史。旧的待提交索引后来获得证据记录，也属于正常追加。分页仅发现新索引，想复查之前的待提交项，应从 `after=0` 重扫或按编号查询；不能认为推进分页游标后会收到旧项的状态更新。

## 初次登记与检查点推进

```python
from embodied_agent.evidence_checkpoint import (
    checkpoint_evidence, discover_checkpointed,
)

# 首次登记是明确的信任起点：调用方先确认数据库来源，再独立保存结果。
initial = checkpoint_evidence(database_path)
if initial["status"] == "checkpointed":
    checkpoint = initial["checkpoint"]

# 重启后使用独立保存的 checkpoint；失败时保留它，不覆盖。
page = discover_checkpointed(database_path, checkpoint)

# 推进时必须传入旧检查点：同一快照中先核对，再生成新版。
advanced = checkpoint_evidence(database_path, previous=checkpoint)
if advanced["status"] == "checkpointed":
    new_checkpoint = advanced["checkpoint"]  # 调用方独立持久保存成功后再替换旧版
```

初次登记只记录当时状态，不证明历史完整或真实。不能在恢复异常时省略 `previous` 重新登记，否则会把缺失历史当作新的基线。读取不修复热日志；需要时应先显式进行 SQLite 可写恢复，再用保留的检查点核对。

## 验证结果

```sh
make evidence-checkpoint-test
make evidence-checkpoint-check EVIDENCE_CHECKPOINT_OUTPUT=outputs/evaluations/evidence-checkpoint-new
```

最终源文件下运行 18 个案例，每种两次，全部符合预期：

| 场景 | 次数 | 结果 |
| --- | --- | --- |
| 原数据库不变 | 2 | 通过，1 条已落账记录 |
| 正常追加 | 2 | 通过，2 条已落账记录 |
| 待提交索引补交记录 | 2 | 通过，1 条已落账记录 |
| 换成另一个数据库 | 2 | 身份不匹配，拒绝 |
| 恢复同身份旧备份 | 2 | 历史不匹配，拒绝 |
| 删除已知证据记录 | 2 | 历史不匹配，拒绝 |
| 删除已知索引 | 2 | 历史不匹配，拒绝 |
| 正文及两表摘要同步改写 | 2 | 历史不匹配，拒绝 |
| 检查点之后追加，再丢失该尾部 | 2 | **仍通过，验证了未覆盖历史的检测限制** |

每次恢复使用新 spawn 进程，仅传入数据库、外部检查点和输出文件路径；输入是合成审计 receipt，证据时钟为虚拟时钟，未运行真实相机采集链。数据库读取前后哈希一致。18/18 指符合上述预期，包含两个预期无法检测的限制案例，不能宣传为全部数据丢失都被检测。

50 项证据链测试通过，其中新增 14 项覆盖身份、旧副本、历史删除、同步改写、只读、分页、追加、旧检查点推进及 WAL 并发快照一致性。Ruff 检查通过。[最终实验摘要](evaluations/evidence-checkpoint-v2.json)记录源文件、报告、数据库和独立检查点哈希。第一版 18/18 原始实验保留在 `outputs/evaluations/evidence-checkpoint-v1`；增加同快照检查点推进接口后，用最终源文件重新执行并保留 v2。

## 实际边界与成本

- 必须保留较新的独立检查点。检查点也回退、丢失或被同步改写时不能检测；检查点之后的新记录未受该检查点保护。这不是签名认证、远端见证或防恶意篡改系统。
- 完整复制同一数据库、保持历史不变，可以通过；身份不是物理文件或设备身份证明。`history_mismatch` 只报告不一致，不断言原因必然是备份回退。
- 检查点验证需要扫描其覆盖的两表历史，耗时与历史大小相关，每页重新验证；分页输出有界不等于校验耗时有界。这是离线恢复检查，不放入 20 ms 证据消费路径，不承诺实时性能。
- 证据前缀使用 SQLite rowid；删除、重建表、可能改变 rowid 的 VACUUM 等维护可能使旧检查点失效，需要单独设计受控迁移。本轮未实现保留清理或大规模增量校验。
- 默认旧接口仍可用；必须主动使用新接口才有检查点保证。整个功能不重放证据、不续期、不授权机器人动作，没有真实机器人或断电耐久性结论。
