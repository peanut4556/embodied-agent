# 带检查点保护的任务执行入口

日期：2026-11-24。

## 本次解决的问题

此前显式 load/run/save 中，任务已经改变暂停状态但保存尚未完成时进程死亡，磁盘可能只剩旧状态。本次新增 run_checkpointed，将协作锁、严格加载、预写保守状态、运行和最终保存组成一个入口。原 AgentRuntime.run 保持不变；只有明确采用该入口或 CLI 参数才启用保护。

执行顺序：

1. 校验任务期限和停止回调；对规范化路径对应的固定 .lock 侧文件获取非阻塞独占锁。
2. 严格加载检查点；仅允许 quarantine 策略，不擅自把 fallback 改为隔离模式。
3. 在磁盘预写保守快照，将 locate/pick/place/verify 以及已知估计动作标为退化。此写入失败时运行时不会被调用。
4. 使用原始加载预算执行任务，内存保留原有隔离与批次约束。
5. 运行正常返回后保存实际最终预算，保存成功才把 TaskResult 返回给调用方。任务失败也正常保存其最终退化状态。
6. 抛异常或进程死亡时不自动清除预写状态；后续进程看到保守暂停。锁由操作系统在进程结束时释放。

这里的预写状态表达“这次运行结果可能不确定”，不代表所有动作真实发生了漂移。因此异常退出后可能比必要范围暂停更多动作。没有根据未知物理状态自动重试或自动解封。

## API 与 CLI

```python
from embodied_agent.checkpointed_runtime import run_checkpointed
result = run_checkpointed(
    runtime, instruction, "budget-state.json", task_timeout_s=2,
)
```

预算文件必须事先明确创建，并使用 quarantine：

```python
from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import save_budget
save_budget(ActionBudget(calibration_samples, drift_policy="quarantine"), "budget-state.json")
```

初始化只能用于新的部署状态，不能覆盖损坏或被暂停的现有状态来解封。

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent "red block box" \
  --budget-checkpoint budget-state.json --task-timeout 5 --json
```

CLI 要求同时设置 --task-timeout。正常成功和任务失败保留既有 0/1 退出码及 JSON 结果。锁冲突、加载/保存错误仍作为异常退出，不输出伪造的成功 TaskResult，也尚未统一为 JSON 错误信封。未提供新参数时完全走原路径。

## 并发和维护

使用 POSIX flock，适用当前 macOS/Linux 本地环境。固定侧文件不得在运行中删除或替换，否则可能形成不同锁域；文件残留不等于锁仍被持有，不应靠删除锁文件“解锁”。该锁仅协调使用同一规范路径与此协议的调用方，不能约束直接调用 save_budget 的程序、硬链接别名或其他机器，不提供网络文件系统一致性保证。

重新校准维护应使用同一锁：

```python
from embodied_agent.checkpointed_runtime import checkpoint_lock
from embodied_agent.budget_checkpoint import load_budget, save_budget
with checkpoint_lock("budget-state.json") as path:
    current = load_budget(path)
    replacement = current.validated_recalibrated(
        "verify", calibration, batch_id="cal-004",
        validation_samples=holdout, validation_batch_id="val-004",
    )
    save_budget(replacement, path)
```

只读检查命令不获取执行锁，运行期间可能看到临时保守快照；它不应被解释为永久故障结论。维护操作则必须获取锁。当前没有自动恢复、租约跨主机协调或抢占挂起任务。

## 验证与限制

新增 10 项测试覆盖正常提交、真实耗时越界状态持久化、再次加载后零派发；真实子进程 os._exit(23) 后保守快照保留及锁释放；预写失败不调用运行时；最终保存失败不返回成功；运行时异常保留不确定状态；跨进程锁冲突不写文件；非法配置/缺失文件；CLI 成功运行及缺失期限；fallback 策略拒绝且不改文件。连同相关回归共 113 项通过。

进程死亡测试证明的是软件进程退出情形，不是断电：沿用原子替换与文件 fsync，未增加目录 fsync 或存储硬件持久性保证。动作可能已经执行但最终结果未提交，不具备物理动作 exactly-once 或自动回滚能力。检查点摘要不是防篡改签名；可信本地路径和合作调用者仍是前提。任务期限从内部 run 开始，检查点 I/O 不包含在该期限内，文件 I/O 也无硬实时上界。该入口不保证设备停止，也未做实机实验。

```sh
make checkpointed-runtime-test
```
