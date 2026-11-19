# 同池一致快照对账

日期：2026-11-19。

新增 `WaitBudgetStore.lookup_many(request_ids)`：接受 1–64 个唯一、长度 1–256 的字符串编号，输入非法或重复时在数据库连接前拒绝。一个读事务内核验完整历史、生成所有结果与同一历史检查点，返回顺序对应输入顺序。单条 lookup 复用该实现，返回合同保持不变。

结果含 single_database_snapshot=true、checkpoint、当前 remaining_ns 和逐编号结果。每条历史回执中的 remaining_ns 仍是分配当时余额，不能与顶层当前余额混淆。所有结果均 execution_verified=false，查询不追加分配、不启动采集、不发布检查点。

新增命令使用一个数据库和显式请求编号列表：

```sh
PYTHONPATH=src:. .venv/bin/python scripts/lookup_budget_batch.py \
  --database /path/pool.db --expected-pool-id ORIGINAL_POOL_ID \
  --requests /path/request-ids.json --checkpoint /path/baseline.json
```

request-ids.json 内容例如 `["request-a", "request-b"]`。checkpoint 可选。请求文件上限 65536 字符，基线文件上限 8192 字符。全部查到退出 0，至少一条未找到退出 1；输入、数据库、身份或历史核验失败退出 2/unconfirmed，不能交付部分可信结果。

新增 5 项测试覆盖顺序/历史回执/检查点一致、读取期间另一连接提交、非法批次在连接前拒绝、独立基线失配时整批拒绝、独立命令的 0/1 退出码。并发测试在临时数据库显式开启 WAL，让另一连接在读取旧记录后提交新记录；当前报告仍含零分配的检查点和全部未找到，下一次查询确认新记录已存在。生产默认日志模式未改变。相关预算及恢复回归共 59 项测试通过，Ruff 检查通过。

```sh
make budget-snapshot-lookup-test
```

本轮没有把多池意图清单审计自动改成单池模式：原 audit_manifest 仍是逐条独立快照，新接口和命令用于已经明确属于一个池的请求组。当前结果仅代表事务读取时的状态，返回后其他进程可以继续分配；不是实时冻结或执行授权。独立基线未覆盖的丢失、基线与数据库共同回滚仍不受保护。没有完整采集性能或实机验证。
