# 操作 ID 与重复请求抑制

日期：2026-11-24。

## 功能和流程

上位机可能因网络超时、窗口重复点击或未收到响应而提交相同任务。新增可选 run_operation，在同一检查点协作锁下先登记操作，再执行已建立的检查点保护流程。无 operation_id 的旧调用保持原行为。

同一 ID 首次请求将请求摘要和 pending 回执写入检查点旁的 `<文件名>.operations` 目录，文件名为 ID 的 SHA-256，不把 ID 当路径。只有登记成功才允许进入检查点执行。实际任务正常返回、最终预算保存成功后，回执改为 complete 并保存完整 TaskResult，然后返回。失败 TaskResult 也属于已完成请求，后续重复请求返回该失败结果，不自动重试。

同 ID、相同请求且 complete：返回历史结果，operation.replayed=true，不调用运行时，也不重写预算文件。同 ID、不同请求摘要：返回 operation_conflict。同 ID 为 pending：返回 operation_in_doubt，不自动恢复执行。损坏或不合法回执同样拒绝，而不是按“没有记录”处理。

因此执行异常、进程直接退出、任务结果已形成但回执保存失败，都可能留下 pending。保守拒绝也可能发生在实际尚未派发动作时，例如登记后加载预算失败；本实现宁可标记不确定，不推测是否安全重试。

## 请求绑定

摘要绑定原始指令、任务期限和 request_context 字典。API 调用方必须把影响任务语义的配置、策略/规划器版本、必要的部署身份放进 context；库无法推断任意 runtime 的内部差异。CLI 自动绑定 planner、确认与观测参数，排除 JSON 显示选项。程序升级或不同外部模型服务的语义变化不会自动被 CLI 检测，操作 ID 必须按部署约定使用，不能把旧历史结果当成新版本重新执行。

```python
from embodied_agent.operation_receipt import run_operation
outcome = run_operation(
    runtime, instruction, "budget-state.json",
    operation_id="pick-request-20261124-001",
    task_timeout_s=2,
    request_context={"planner_version": "rule-v1", "policy_version": "policy-v1"},
)
```

ID 非空、无首尾空白、最多 128 字符。作用域是同一规范化检查点路径。相同 ID 使用另一个检查点不受本机制关联。超时值按浮点秒规范化。摘要不保存完整原指令，但 TaskResult 的消息可能含任务信息，回执仍按本地业务数据管理。

## CLI 与结果

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent "red block box" \
  --budget-checkpoint budget-state.json --task-timeout 5 \
  --operation-id pick-request-001 --json
```

--operation-id 必须配合 --budget-checkpoint。完成结果保留现有任务字段，新增 operation.id/replayed。重放结果 checkpoint.phase=historical，checkpoint_state=historical_committed，明确说明是历史结果，不证明当前设备状态或当前预算仍相同。历史成功退出 0，历史失败退出 1；冲突、不确定或其他执行流程错误退出 3；参数解析错误为 2。

execution_started 始终描述本次调用是否进入运行时。重复 pending 的本次调用未进入运行时，但 error.action_outcome_unknown=true，因为原调用可能执行过动作。不能用 execution_started=false 推断该 ID 从未执行。

## 文件与一致性边界

回执写入使用同目录临时文件、文件 fsync 和原子替换；严格 JSON 读取与最大 1 Mi 字符限制沿用检查点工具，并验证规范化 SHA-256、schema 和结果基本类型。摘要防非恶意损坏，不是签名，无法阻止能重算摘要的篡改。目录与文件须位于可信本地路径。

操作登记、预算提交和回执完成不是一个跨文件原子事务。崩溃可能造成保守 pending，正常流程不会据此重新执行。进程死亡测试不等于断电持久性验证。协议仍依赖同一 POSIX 协作锁；不协调绕过协议的写入、其他检查点、其他机器或恶意进程。

回执不自动过期或删除，以免删除后相同 ID 重新被接受。当前没有自动清理工具或容量回收策略，部署方需监控目录增长；删除回执会失去该 ID 的重复抑制证据。没有自动解 pending 的 API，也不能通过换 ID 盲目重试不确定物理动作，应先外部核对设备状态。

这是合作调用方的重复请求抑制，不是物理动作 exactly-once。操作可能已执行但响应丢失，重放仅返回历史记录；机制不能回滚物理动作，也不判断目标物体是否仍在原位置。旧 run_checkpointed 和直接 runtime.run 不受 ID 保护。

## 测试

新增 10 项测试覆盖完成结果重复返回且只调用一次、参数冲突、失败任务重放、执行异常拒绝重试、回执完成写入失败、登记失败零执行、损坏回执拒绝、真实进程 os._exit 后拒绝重试、跨 CLI 进程重放、非法 ID。相关回归共 133 项通过，源码检查通过。未增加实机实验或修改既有预算实验数据。

```sh
make operation-receipt-test
```
