# 批量恢复审计

日期：2026-11-19。

新增 `budget_recovery_audit.audit_manifest()` 和命令 `scripts/audit_budget_recovery.py`，把单条意图对账扩展为显式清单驱动的只读批量检查。

清单示例：

```json
{"version": 1, "entries": [
  {"intent": "attempt-001.json", "checkpoint": "baseline.json"},
  {"intent": "attempt-002.json"}
]}
```

```sh
PYTHONPATH=src:. .venv/bin/python scripts/audit_budget_recovery.py --manifest /path/manifest.json
```

相对路径以清单目录为基准，不受启动命令当前目录影响；绝对路径可用。整个清单先验证再读取意图：版本必须为 1，条目 1–64 个，只允许 intent/checkpoint 字段，解析到相同路径的重复意图拒绝。清单最多读取 65537 字符并拒绝超过 65536 字符，单个检查点超过 8192 字符拒绝。数量限制是最多 64 个文件条目，不是时间或内存硬实时保证；不同文件中重复的请求身份不会自动合并，统计表示文件结果数。

每条返回原对账状态及路径，缺失或损坏文件、数据库错误、身份或历史不匹配单独返回 unconfirmed，继续下一条。指定检查点但无法读取时不退回无检查点模式。输出含计数、检查条数及退出码：全部查到为 0；有未找到且无无法确认为 1；任何无法确认为 2。清单本身非法也退出 2，不查询任何意图。

每条是独立读取快照，不保证所有条目在同一时刻观察数据库，明确输出 single_database_snapshot=false。execution_verified 始终为 false；all_allocations_found 仅表示每个清单条目均查到额度，不表示采集完成。工具不写预算、不发布基线、不启动采集、不自动重试。

新增 5 项测试验证混合成功/损坏/未找到结果保持及数据库不变、重复/超量清单在查询前拒绝、检查点缺失不降级、独立 CLI 的 0/1 退出码、空及超长清单拒绝。与既有恢复和预算回归共 54 项测试通过，Ruff 检查通过。

```sh
make budget-recovery-audit-test
```

清单、意图与检查点需可信保管。无基线或基线未覆盖的回滚限制仍在，多个文件也可能共同被替换。本轮提供批量运维能力，没有新增完整采集性能、真实相机或机器人验证。
