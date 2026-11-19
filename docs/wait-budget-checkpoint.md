# 预算历史检查点

日期：2026-11-19。

本轮新增 `WaitBudgetStore.checkpoint()` 和可选 `expected_checkpoint` 参数。检查点包含版本、池编号、已分配槽位数及 SHA-256 摘要；摘要覆盖池身份、总阈值、槽位上限和有序分配记录。导出使用同一读事务，不领取额度。

```python
baseline = pool.checkpoint()  # 由调用方保存到独立、可信的位置
# 后续启动读取独立保存的 baseline，而不是重新从待检查数据库生成。
pool = WaitBudgetStore(path, expected_checkpoint=baseline)
```

打开时验证历史至少包含基线记录数，并对该前缀重新计算摘要。每次 `reserve()` 的 `BEGIN IMMEDIATE` 事务内也重新核对，因此打开后数据库发生变化，不能靠旧检查结果继续分配。正常追加不会使前缀失效。`snapshot()` 和再次导出检查点同样遵守已有基线。

输入检查点严格检查字段、版本、计数和摘要格式，保存深拷贝，调用方修改原字典不会改变已经绑定的基线。已有 `expected_pool_id` 可以同时使用。默认不传检查点保持兼容。

四项新增测试使用真实临时 SQLite 文件：恢复同池旧副本后拒绝打开；合法追加和输入字典隔离；保持记录数量但修改旧请求编号后拒绝继续分配，数据库行数不变；配置改动及非法检查点拒绝。与池身份、持久化额度和恢复测试合计 27 项通过，Ruff 检查通过。

```sh
make wait-budget-checkpoint-test
PYTHONPATH=src:.:tests/physics .venv/bin/python -m unittest test_wait_budget_checkpoint test_wait_budget_identity test_wait_budget_store test_recovery_wait_budget test_capture_recovery
```

这是带独立历史基线的准入校验，不是通用防回滚系统。调用方负责基线保存、信任来源和更新时机；本轮未实现自动基线发布、跨文件原子提交或签名认证。基线之后尚未被覆盖的追加记录若丢失，旧基线不能发现；数据库和基线同时回退也无法发现。不能从不可信数据库即时导出基线再用它自证。

本轮没有重新跑完整 36 请求采集实验，不以之前的实验结果替代当前版本验证。该功能不授予设备所有权、不保证真实硬件安全，也不是硬 CPU 配额。
