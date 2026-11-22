# 见证回调的进程隔离

日期：2026-11-22。

## 问题与实现

第二路见证可以拒绝冲突观测，但同步回调若永久阻塞，原有确认期限就没有机会再次检查。新增可选 `ProcessWitness`：在 spawn 子进程内创建并调用见证读取器，父进程等待响应就绪。等待超时后关闭该实例，先 terminate，等待 100 ms；仍存活则 kill，再等待 100 ms。确认退出后记录 exitcode 并释放进程句柄。重复 close 安全；回收未确认时 close 返回 False，调用者可再次检查。

`AgentRuntime` 将 `WitnessTimeout` 转为 `witness_timeout`，不接受单源成功，不自动重发请求；报告耗时包含超时等待和回收。`witness_reads` 仍表示成功返回的见证读取次数，超时的首次读取为 0。初始化失败继续抛出；进程读取异常通过下述结构化故障处理，不伪装成观测反例。

每个实例最多 128 次请求，达到上限后拒绝继续调用。超时后必须显式创建新实例，避免悄悄重试或复用失效通信通道。默认响应等待 50 ms，启动等待 5 s，均可配置在 (0, 5] 秒范围内。

## 接入方式

在可导入的 Python 模块顶层定义 factory，由 factory 在子进程中打开见证资源，并返回接收 Action、返回 Observation 的 callable；不要把父进程已打开的设备句柄传入。factory 必须能被 spawn 序列化。程序入口使用 `if __name__ == "__main__":` 保护。

```python
from embodied_agent.process_witness import ProcessWitness
from embodied_agent.runtime import AgentRuntime

# create_witness_reader 是应用提供的顶层工厂。
with ProcessWitness(create_witness_reader, timeout_s=0.05) as witness:
    runtime = AgentRuntime(
        planner, policy, robot,
        confirmation_timeout_s=0.5,
        max_observation_age_s=0.1,
        confirmation_witness=witness,
    )
    result = runtime.run(instruction)
```

见证 Observation 仍须带有与父进程同一时钟域的 `observed_at_monotonic`，通过原有新鲜度、两源时间差及结果检查。上下文管理器负责 close；若需要判断回收是否确认，显式检查 close 的返回值和 exitcode。当前没有虚构第二传感器的 CLI 开关。

## 验证

新增 4 项真实进程测试：正常读取及重复关闭；无限阻塞且忽略 SIGTERM 后回收并拒绝复用；主源正面观测时仍报告见证超时且计入耗时；非法超时与初始化异常。与主运行时、确认故障矩阵、随机压力和见证压力回归合计 53 项通过。

```sh
make process-witness-test
```

这次未重新生成或修改已冻结的实验报告，也没有实机批次、训练或校正。

## 边界

这是可信本地回调的进程隔离，不是安全沙箱。Pipe 使用 Python 序列化，只应连接可信工作进程。计时覆盖响应就绪等待，未对 IPC 大载荷发送、完整接收或操作系统调度建立硬实时上界；确认总期限可能被一次等待及回收超出。主 observe 和 execute 仍可能阻塞。

强制杀进程不保证设备释放或物理动作停止；任意父进程死亡后的回收也不在本实现保证范围内。本模块未继承 CaptureSupervisor 的独立监督能力。第二源真实性、物理独立性和共同错误风险仍需单独验证。

## 扩展：工作进程故障闭环（2026-11-22）

新增 `WitnessFailure`：通信断开对应 `witness_disconnected`，工作进程抛异常或返回非 Observation 对应 `witness_read_failed`，关闭实例与次数耗尽分别对应 `witness_closed`、`witness_request_limit`。运行时返回失败 TaskResult，报告 status 为 `witness_failed`，并在 witness_failure 中保留原因。超时仍独立返回 witness_timeout。普通自定义回调的任意异常不会被吞掉，初始化异常也仍交给调用方。

真实 spawn 子进程测试分别注入 os._exit(23)、非法返回值和包含私有设备信息的异常。通过完整 run 路径验证：原本两步的计划只执行一次动作、完成步数为零、停止后续动作、报告原因且不包含私有异常文本、进程已回收。与既有回归合计 54 项测试通过。这里的“停止”指不再发出后续动作，不证明正在发生的物理运动已经停止。原有 IPC、硬实时及父进程死亡边界保持不变。
