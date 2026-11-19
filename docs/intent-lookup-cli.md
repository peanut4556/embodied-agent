# 意图文件驱动的恢复对账命令

日期：2026-11-19。

`lookup_wait_budget.py` 新增 `--intent` 模式，恢复操作者无需手工复制数据库路径、池身份或会话编号。三者从此前持久化的可信意图文件读取。原有三参数模式保持兼容；两种模式禁止混用，参数不完整或混用时 argparse 拒绝，不查询也不分配。

```sh
PYTHONPATH=src:. .venv/bin/python scripts/lookup_wait_budget.py \
  --intent /absolute/path/attempt-001.json \
  --checkpoint /absolute/path/independent-baseline.json
```

`--checkpoint` 可选，API `reconcile_intent(path, expected_checkpoint=...)` 同步支持历史基线。应使用独立保存、覆盖已知分配的基线，不能从待检查的库即时生成来证明自身正确。核验失败输出 JSON `unconfirmed`，退出码 2。

结果语义保持一致：0 表示查到历史分配，1 表示读取快照中未找到，2 表示无法确认或命令使用错误（参数错误走 stderr，业务错误返回 JSON）。查询不分配、不启动采集、不更新基线。查到记录仍不代表任务已经执行；未找到也不是自动重试许可。

新增五项测试使用真实临时数据库和独立命令进程：子进程写意图并提交后以 73 退出，新进程仅凭路径查到原分配；意图存在但未提交时返回 1 且记录数为零；恢复同池旧副本后携带独立基线返回 2/unconfirmed；意图缺失或损坏拒绝；混入手填编号拒绝。与相关预算及恢复回归共 49 项测试通过，Ruff 检查通过。

```sh
make intent-lookup-cli-test
```

本轮推进的是可操作恢复入口和错误分流，没有新的完整采集性能或真实设备实验。没有检查点时，同池回滚仍可能表现为 not_found；基线未覆盖的历史丢失、数据库与基线同时回滚仍不受保护。意图和基线均非签名材料，应可信保管；此命令不承担设备所有权恢复。
