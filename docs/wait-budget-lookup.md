# 预算分配结果只读对账

日期：2026-11-19。

新增 `WaitBudgetStore.lookup(request_id)` 与 `scripts/lookup_wait_budget.py`。查询在同一读事务内校验池配置、历史合同及可选独立检查点，再查找原请求编号；不追加分配、不刷新基线、不启动采集。

找到时返回 `status=reserved` 和原分配回执。回执的 `remaining_ns` 是当时分配后的余额，通过此前各槽位的固定份额重建，不是当前余额；后续合法追加不会改变它。当前余额仍用 `snapshot()` 查看。未找到返回 `not_found`，两者均明确 `execution_verified=false`。

```sh
PYTHONPATH=src:. .venv/bin/python scripts/lookup_wait_budget.py \
  --database /absolute/path/budget.db \
  --expected-pool-id ORIGINAL_POOL_ID \
  --request-id ORIGINAL_REQUEST_ID
```

命令必须提供独立保存的预期池编号，可另传 `--checkpoint /path/baseline.json`。退出码 0 表示查到历史分配，1 表示该读取快照中未找到，2 表示数据库/配置/检查点/身份等错误导致无法确认，输出 JSON。参数语法错误由 argparse 按标准错误入口处理。不要把退出码 2 当成未提交，不要据此自动重试。结果不会凭空恢复原请求编号；恢复管理器的编号是会话 UUID，外部调用方需事先保留或从已有报告获知，自动发现仍可使用现有 snapshot。

新增 5 项测试验证历史回执在后续追加后不变、查询不占额度、实际子进程提交后 `os._exit(73)` 再由独立进程查询、错误池及损坏检查点返回无法确认、非法请求编号拒绝。与精确分配、历史基线、持久化池和恢复回归合计 40 项测试通过，Ruff 检查通过。

```sh
make wait-budget-lookup-test
```

本轮实际退出发生在 reserve 已提交返回后、外部父进程尚未收到回执时；没有模拟掉电或磁盘损坏。只读指不执行分配或应用数据写入，SQLite 打开和恢复自身日志的文件系统行为仍遵循其机制。

历史分配只证明份额已记账，不能证明采集后端创建、硬件动作发生或任务完成，也不是再次执行许可。未找到只是查询快照结论，并不排除后续其他进程提交。数据库与基线同时回退、独立基线未覆盖的丢失记录等既有限制仍在。本轮未重新跑完整采集性能实验。
