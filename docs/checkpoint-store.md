# 可事务更新的独立检查点仓库

2026-11-14。本轮补上上一轮“调用方自行保存检查点”的缺口：新增独立 SQLite 检查点仓库，提供显式登记、版本比较更新、完整性核对和重启后的只读恢复。更新进程崩溃后，可以检查持久化版本，不依赖更新回执是否送达。

## 接口与事务合同

```python
from embodied_agent.checkpoint_store import (
    CheckpointStore, read_checkpoint_store, discover_from_store,
)

# 仅在确认数据库来源后的首次登记使用；不是恢复失败的自动处理分支。
with CheckpointStore(checkpoint_path, create=True) as store:
    enrolled = store.enroll(evidence_path)

# 日常更新：缺失仓库不会被默认创建。
head = read_checkpoint_store(checkpoint_path)
if head['status'] == 'loaded':
    with CheckpointStore(checkpoint_path) as store:
        publication = store.advance(
            evidence_path, expected_generation=head['generation'],
        )

# 重启后无需自行拼装 JSON 检查点。
page = discover_from_store(evidence_path, checkpoint_path)
```

仓库只保留一个当前检查点及其 generation，不是无限增长的发布日志。版本号、规范 JSON 检查点共同计算 SHA-256；篡改版本号或正文而未同步改写摘要，读取会报告 `integrity_error`。

更新在仓库 `BEGIN IMMEDIATE` 事务内完成：读取旧版本，比较调用方期望版本，调用上一轮接口核验源库历史并捕获新检查点，最后提交新版本。仓库锁覆盖整个更新，其他发布者不能在核验与提交之间替换检查点。SQLite 锁等待为零，竞争失败显式返回，不阻塞重试。

| 状态 | 含义 |
| --- | --- |
| `published` | 本次更新提交完成，返回新 generation |
| `already_enrolled` | 已登记；即使显式调用 enroll 也不能覆盖已有基线 |
| `generation_conflict` | 调用方版本过期，未写入新的检查点 |
| `unenrolled` | 仓库存在但无基线；advance 不自动登记 |
| `source_rejected` | 源库身份、历史或可读性核验失败，旧检查点保持不变 |
| `integrity_error` | 仓库检查点或版本完整性核验失败，禁止读取为正常基线或继续更新 |
| `publication_unconfirmed` | SQLite 操作或提交异常，不能据此推断更新一定没发生；应重新读取核对 |

`read_checkpoint_store` 和 `discover_from_store` 只打开已有仓库；缺失返回 `unavailable`，不创建文件。恢复结果带 `checkpoint_generation`，表示本次仓库读事务观察到的版本。仓库读事务覆盖源库发现过程，但这不是两个数据库的分布式原子快照，也不包含稍后发生的发布。

源库与仓库不能是同一个路径、符号链接目标或硬链接文件。独立文件只是最低边界；部署时仍需把仓库保存在不会随源库一起回退的位置。本轮没有实现远端见证或独立存储服务。

## 故障实验与回归

```sh
make checkpoint-store-test
make checkpoint-store-check CHECKPOINT_STORE_OUTPUT=outputs/evaluations/checkpoint-store-new
```

先写入第一条证据并登记 generation=1，再追加第二条证据；独立 spawn 进程尝试更新到 generation=2。使用合成审计 receipt、虚拟证据时钟、真实 SQLite 文件和真实 `os._exit` 退出，未运行相机链。

| 场景（每种 2 次） | 退出码 | 回执 | 重启读取 |
| --- | --- | --- | --- |
| 正常更新 | 0 | 有 | generation=2，检查点覆盖 2 条记录 |
| UPDATE 完成、COMMIT 前退出 | 81 | 无 | generation=1，检查点仍覆盖 1 条记录 |
| COMMIT 后、回执前退出 | 82 | 无 | generation=2，检查点覆盖 2 条记录 |

6/6 符合预期，读取到的检查点 JSON 和摘要均完整；恢复代码只打开可写连接让 SQLite 处理热日志，再读仓库，不调用 enroll 或 advance。三类恢复都可发现源库已有的两条记录，但提交前退出时第二条尚未受持久化检查点覆盖，不能混淆“发现”和“已检查点保护”。

12 项新增仓库测试及 50 项原证据链回归通过，共 62 项相关测试。覆盖 stale generation 拒绝、锁竞争不确定结果、重复登记、空仓库、损坏、版本损坏、路径别名、源库丢失历史、缺失文件及只读字节不变。Ruff 检查通过。

[最终实验摘要](evaluations/checkpoint-store-v2.json)含源代码、原始报告、6 个源数据库与 6 个检查点仓库的 SHA-256。第一轮结果保留于 `outputs/evaluations/checkpoint-store-v1`；增加版本号摘要绑定后，使用最终源文件重新执行并保存 v2，两轮均为 6/6。

## 能力边界

- 核心进展是把检查点发布变成可对账的持久化事务，并拒绝过期发布者覆盖新版本。不是机器人动作许可，不重放证据或刷新有效期。
- 更新时锁住仓库并扫描源库已有历史；开销随历史增长，适合离线恢复，不进入 20 ms 消费路径，尚未验证大规模吞吐或长时间锁竞争。
- 仓库与源库同时回退，或仓库本身回退且无更外部版本见证，仍可能无法检测。generation 是本地仓库版本，不是不可回退的硬件计数器。
- SHA-256 是一致性校验，不是签名。能同时改写正文、版本和摘要的攻击者不在此保护范围内。
- 实验覆盖进程退出，不覆盖断电、文件系统故障或硬件持久化保证。发布异常先核对仓库，不应盲目重试；热日志可能需要显式可写恢复。
- 首次 enroll 是信任决策。仓库缺失或损坏时，不能把重新创建并登记作为自动恢复路径。默认旧接口未改变，用户须主动使用本适配器。
