# 独立时限后的会话重建验证

2026-11-16。本轮新增真实进程恢复实验，将独立监督运行时限与 CaptureRecovery 会话隔离串联。实验要求旧初始化永久阻塞时独立超时回收，再显式创建新会话完成确认；所有失败原样保留。

## 流程

```sh
make supervised-recovery-check SUPERVISED_RECOVERY_OUTPUT=outputs/evaluations/supervised-recovery-new
make supervised-recovery-test
```

管理器设置 supervised=True、reserved 采集配置、runtime_limit_s=4、max_generations=2。首次 factory 忽略 SIGTERM 并永久 sleep，初始等待上限仍为 5 秒；独立监督应先触发 4 秒作业限制并回收读取进程。旧进程清理未核实时，不允许创建新后端。

随后显式使用快速合成读取器建立新会话，以旧 session 尝试领取证据，再运行完整预约确认、领取和重复领取检查。最后关闭全部资源，尝试第三次启动，验证次数预算拒绝。没有自动重试、跳过确认或延长曝光窗口。

## 首轮结果：隔离通过，完整确认未通过

两次实验均完成以下检查：

- 第一轮初始化超时，至少一个监督报告 runtime_limit_exceeded=true；另一槽位可能随取消一同回收。
- 旧监督进程和读取进程全部核实退出，之后才创建不同编号的新会话。
- 旧令牌领取新证据返回 session_mismatch。
- 新会话最终也完成资源回收，第三次启动被 generation budget exhausted 拒绝。

但两次新会话都没有完成最终确认：

| 运行 | 请求数 | 读取器错误 | 控制器结论 |
| --- | --- | --- | --- |
| 1 | 5 | scheduled_capture_window_missed | reservation_exposure_rejected |
| 2 | 15 | scheduled_capture_window_missed | reservation_exposure_rejected |

请求记录显示目标曝光到最晚曝光之间仍为原有 2 ms 窗口；这两次未能满足该窗口。现有记录能定位为曝光时序拒绝，不能进一步断言具体 OS 调度或 IPC 原因。没有放宽窗口，也没有为获得通过结果而再次运行。

因此端到端目标通过数为 **0/2**，不能宣传为恢复后完整确认成功；恢复资源隔离和会话隔离检查为 **2/2**。脚本写出完整报告并返回非零退出码。新会话没有产生可领取证据，没有实机动作。

[实验摘要](evaluations/supervised-recovery-v1.json)记录源码哈希、原始报告路径/哈希、旧清理状态、旧令牌拒绝和次数限制结果；完整报告保存每个请求及 driver_errors。新增 1 项实际进程单元测试验证短时初始化故障后的显式重建与旧令牌拒绝；原恢复 9 项、时限 3 项、监督采集 3 项测试通过，共 16 项相关测试。Ruff 检查通过。

## 进展与下一瓶颈

新增的是可重复运行的“独立超时→确认旧进程退出→新会话→任务确认”恢复验证工具。本轮证据明确：资源与会话恢复机制有效，不代表恢复后采集调度一定满足严格窗口。后续优化需要针对恢复后的曝光调度做分段测量，不能靠自动重试或扩大容差掩盖失败。

读取器是合成双视角，时钟和进程为真实运行。初始化阻塞会释放 GIL；没有真实相机、硬件恢复、断电或长期成功率结论。
