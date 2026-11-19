# 绑定已查看状态的预算分配

日期：2026-11-19。

新增 `reserve(request_id, reviewed_checkpoint=...)`。调用方可以先导出检查点，查看状态后再携带原检查点领取额度。分配在 `BEGIN IMMEDIATE` 事务内重新核对完整配置及全部历史，只有完全匹配才插入新记录并提交。合法追加也会使旧检查点过期，即使池内仍有余额。

```python
review = pool.checkpoint()
# 调用方查看并决定继续；不得在提交时偷偷替换为最新检查点。
receipt = pool.reserve("request-unique-id", reviewed_checkpoint=review)
```

`expected_checkpoint` 仍是历史前缀约束，允许后续合法追加；`reviewed_checkpoint` 是本次操作的精确状态条件。两者可以同时使用，各自在事务中执行。默认省略 reviewed 参数保持旧分配行为。检查点输入复用严格合同并深拷贝，非法输入不写入数据库。

新增四项测试：已有合法追加后旧快照拒绝、重新查看后成功；前缀基线与精确状态语义区分；非法检查点不分配；两个真实 spawn 进程共享同一初始检查点并发申请。并发测试使用三个槽位，最终恰好一条分配，另一进程以 `reviewed budget checkpoint is stale` 拒绝，两进程正常退出；因此不是因为额度耗尽才阻止第二次申请。

与历史基线、池身份、持久化额度和恢复回归合计 31 项测试通过，Ruff 检查通过。

```sh
make reviewed-wait-budget-test
PYTHONPATH=src:.:tests/physics .venv/bin/python -m unittest test_reviewed_wait_budget test_wait_budget_checkpoint test_wait_budget_identity test_wait_budget_store test_recovery_wait_budget test_capture_recovery
```

检查点只绑定预算池状态，不绑定审批人或请求意图，也不是签名、授权令牌或跨数据库事务。提交后回执丢失仍可能已消耗额度，不应凭错误自动重放；现有 snapshot 可用于只读查看历史。本轮没有把精确分配选项自动接入 CaptureRecovery，也没有重新测量完整采集性能。数据库与独立基线一起回退的问题仍存在。
