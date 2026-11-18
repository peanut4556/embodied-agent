# 持久化等待预算池

日期：2026-11-18。

本轮把恢复预算从管理器内存扩展到显式创建的 SQLite 预算池。多个管理器或进程重开同一个池时共享已分配记录，避免应用重建后重新领取完整阈值。默认不使用持久化池。

## 合同与实现

`WaitBudgetStore.create(path, total_threshold_ns=..., slots=...)` 只创建新文件，已存在时拒绝。阈值使用整数纳秒，范围 0–1 秒，槽位数 1–3。份额均分，除不尽的整数余数分配给前面的槽位，所有份额之和精确等于总阈值。零阈值仍消耗槽位。

`WaitBudgetStore(path)` 只打开现存数据库；缺失、损坏或配置不合法时不自动重建。配置包含随机池身份，已打开对象每次事务核对身份和配置，并检查已有份额顺序及金额。

`reserve(request_id)` 使用 `BEGIN IMMEDIATE` 串行化读取与插入，`synchronous=FULL`，成功提交后才返回份额。唯一请求编号不能再次领取，即使重复调用也拒绝；没有退款、复用、自动续期接口。锁等待 250 ms 后失败。`snapshot()` 在同一读事务中返回当前记录，可在回执缺失后查询历史，但不能据此再次启动已预留请求。

恢复管理器新增 `persistent_wait_budget` 参数，和本地 `wait_cpu_budget_s` 互斥，仅允许 reserved 模式。通过旧资源清理核验后，用新会话编号申请份额，提交成功才构造采集后端。分配失败或提交结果不明确时不启动后端，锁定当前管理器的启动入口；`close()` 不清除该拒绝状态。新管理器也必须重新读取同一池，已提交份额不会自动退回。构造采集后端失败依旧保留份额并锁定所有权异常。

池槽位数控制跨管理器可分配次数；`max_generations` 仍单独限制一个管理器内的代数。结果保留池编号和最后一次持久化回执，该回执中的余额是提交时的值，不是其他进程继续分配后的实时余额；查看当前余额应调用池的 `snapshot()`。原来的本地阈值和未分配字段在持久化模式下为 `None`。

```python
from embodied_agent.wait_budget_store import WaitBudgetStore
from embodied_agent.capture_recovery import CaptureRecovery

# 首次配置时执行一次；后续启动只调用 WaitBudgetStore(path)。
pool = WaitBudgetStore.create("wait-budget.db", total_threshold_ns=10_000_000, slots=2)
manager = CaptureRecovery(
    position_error_m=(0.001,) * 3,
    capture_profile="reserved",
    spin_window_s=0.0015,
    persistent_wait_budget=pool,
)
```

## 实际进程实验

| 场景 | 结果 |
|---|---|
| 4 个独立 spawn 进程争用 2 个槽位、总阈值 10 ms | 恰好 2 个各获配 5 ms，另外 2 个因耗尽拒绝；4 个进程退出码均为 0 |
| 子进程提交唯一 5 ms 份额后，对外报告前 `os._exit(73)` | 退出码 73；新对象重开池仍见 1 个已分配槽位，换新请求编号也因耗尽拒绝 |

上述实验没有启动相机或机器人，验证的是实际 SQLite 事务和 OS 进程行为，不是设备恢复。没有模拟掉电、磁盘故障或提交系统调用中断；第二项明确是在 `reserve()` 已返回后退出，但外部父进程没有收到分配回执。

测试另外使用实际采集子进程验证第一管理器初始化失败并清理后，新管理器只能获得第二份额度；构造异常及池耗尽测试核对了不退款和拒绝时不调用后端。

## 边界

- 这是回退阈值的持久化分配，不是实际 CPU 时间的硬配额；等待请求可越过阈值，休眠、初始化和其他线程仍会消耗 CPU。
- 预算池不授予设备所有权。多个进程获配不代表允许同时操作同一硬件；应用重启后还需独立确认旧采集作业是否退出。本轮没有实现跨进程设备租约或自动恢复旧会话。
- 保证依赖同一数据库及正常本地文件系统语义。删除、恢复旧副本、替换数据库或另建新池会绕过历史，当前没有签名、外部见证或防回滚机制。
- 提交成功但没能启动会浪费份额，这是保守设计。初始化中断可能留下不可用文件，后续拒绝打开，不自动清空修复。
- 没有全链路持久化模式的 36 请求成功率结论；上一轮完整恢复实验不能替代本轮模式验证。

## 验证与复现

55 项相关测试通过，含新增 6 项：重开/精确整数分配/重复拒绝、创建与输入合同、锁竞争、管理器重建与启动前拒绝、真实失败启动后重建、身份变化与参数互斥。既有预算、预约时序、进程、恢复及监督恢复回归通过。Ruff 检查及实验源码哈希复核通过。

```sh
make persistent-wait-budget-check PERSISTENT_WAIT_BUDGET_OUTPUT=outputs/evaluations/persistent-wait-budget-new
make wait-budget-store-test
```

[提交的实验摘要](evaluations/persistent-wait-budget-v1.json)保留进程争用结果、退出码、持久化快照及源码哈希。两份实际数据库位于本地忽略目录 `outputs/evaluations/persistent-wait-budget-v1/`，不随 Git 提交。
