# 启动意图持久化与无编号对账

日期：2026-11-19。

本轮为 `CaptureRecovery.start()` 增加可选 `budget_intent_path`。必须配合持久化预算池使用。旧资源清理通过、生成新会话编号后，先独占创建意图文件，保存格式版本、数据库绝对路径、池身份和请求编号；文件 flush/fsync 和父目录 fsync 成功后才申请额度，再创建采集后端。

```python
token = manager.start(factory, budget_intent_path="attempt-001.json")
# 应用退出后，新恢复程序无需从内存取得请求编号：
from embodied_agent.budget_start_intent import reconcile_intent
result = reconcile_intent("attempt-001.json")
```

可同时传 reviewed_budget_checkpoint，精确状态分配门槛仍然生效。意图文件不包含审批凭证，也不是执行许可证。文件已存在、写入或同步失败时，不申请额度、不创建采集后端，并沿用预算分配失败锁定状态。部分写入保留，后续读取拒绝损坏内容，不自动覆盖修复。

`reconcile_intent()` 最多读取 8193 字符，拒绝超过 8192 字符或格式非法的意图，按文件保存的预期池身份打开数据库，再调用只读 lookup。它不会 reserve、重试或启动采集。意图文件应来自可信、独立保管的位置；内容不是签名，数据库路径与池身份被共同替换不在防护范围内。

新增 4 项测试：实际子进程在 reserve 调用前或提交返回后 `os._exit(73)`，恢复端仅凭意图文件分别得到 not_found/reserved，最终只有一条分配；已有意图文件阻止额度和后端创建并保持内容；损坏/超长文件拒绝；无持久化池时不创建意图。与相关额度及恢复回归共 44 项测试通过，Ruff 检查通过。

```sh
make budget-start-intent-test
```

退出测试在真实 OS 进程和 SQLite 上运行，但没有启动真实相机或机器人。未注入掉电或文件系统故障，fsync 不构成跨文件和数据库的原子提交。意图存在只代表曾准备请求，已分配只代表额度记账；二者都不能证明采集启动或任务完成。not_found 是读取快照结论，不排除其他进程之后提交。没有自动续跑或退款，本轮也未重新跑完整 36 请求性能实验。
