# 按已查看预算启动恢复任务

日期：2026-11-19。

本轮把上一轮精确检查点分配接入 `CaptureRecovery`，新增可选 `require_reviewed_budget=True`。启用时必须配置持久化预算池，每次 `start()` 都要传 `reviewed_budget_checkpoint`。

```python
manager = CaptureRecovery(
    position_error_m=(0.001,) * 3,
    capture_profile="reserved",
    persistent_wait_budget=pool,
    require_reviewed_budget=True,
)
review = pool.checkpoint()
# 调用方检查状态后明确提交原快照。
token = manager.start(factory, reviewed_budget_checkpoint=review)
```

管理器严格检查检查点并复制输入，不会替调用方读取最新快照。未提供或输入非法在改变尝试次数、清理旧任务及申请额度之前拒绝。提供检查点却没有持久化池同样拒绝。默认不启用严格模式，兼容原调用；非严格模式也可对单次启动显式传检查点。

合法启动依旧先核验旧资源清理，再在持久化池写事务中比较完整状态并提交份额，最后才创建采集后端。已查看之后发生合法追加会导致过期拒绝，即使还有余额。拒绝后沿用 `budget_reservation_failed` 锁定状态，close 不清除它；重新决策需要新管理器及明确检查点，不能靠自动重试换取额度。结果暴露 `require_reviewed_budget`，便于核对运行模式。

新增四项测试覆盖：缺少检查点不启动、不占次数、不占额度；其他调用方先分配后旧检查点拒绝且后端未构造、close 后锁定保留；真实子进程停止后必须显式提供新检查点才能开始下一会话；非法模式及没有池的检查点拒绝。与既有精确分配、历史基线、身份、持久化池及恢复回归共 35 项测试通过，Ruff 检查通过。

```sh
make reviewed-recovery-test
PYTHONPATH=src:.:tests/physics .venv/bin/python -m unittest test_reviewed_recovery test_reviewed_wait_budget test_wait_budget_checkpoint test_wait_budget_identity test_wait_budget_store test_recovery_wait_budget test_capture_recovery
```

严格模式提供预算状态一致性，不证明检查点经过某个人审批，不认证身份，不授予硬件所有权。调用方如立即自动导出并提交最新检查点，只获得并发比较语义，不能宣称人工审核。本轮未重新运行完整 36 请求确认或 CPU 对照；真实子进程测试只验证启动、停止与代际重建。仍没有真实相机或机器人实验。
