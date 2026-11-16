# 监督进程失联时的读取进程退出保护

2026-11-16。本轮补齐独立监督结构中的另一条故障方向：监督进程被突然杀死，实际读取进程尚在运行。新增监督→读取进程的独立存活连接，在读取进程执行 job/factory 之前启动监听线程；唯一写端由监督进程持有，监督死亡触发 EOF，读取进程 os._exit(87)。

这是双向分工：读取进程暂停或内部线程不能运行时，由独立监督终止它；监督进程死亡且读取进程仍可调度时，由内部监听线程退出。两者不能同时保证任意故障下恢复。

## 状态合同

SupervisedCaptureJob.snapshot 新增 cleanup_status：active、reported_stopped、supervisor_lost。监督进程死亡且没有最终 stopped 报告时返回 supervisor_lost。

已有 ProcessPairCapture(supervised=True) 自动获得该保护，无需额外参数。即使外部 ps 已观察到读取进程退出，内部缺少监督进程提供的退出码与回收报告，workers_stopped 仍为 false。因此恢复管理器不能误把此场景作为已验证清理完成而自动重建。

普通 owner_watchdog 和默认非监督模式未改变。正常任务完成时，不主动关闭监听线程正在读取的连接，由进程退出释放它；初版回归曾暴露主动关闭会被误判为监督失联、产生退出码 87，已修复，正常完成重新验证为退出码 0。

## 真实故障实验

```sh
make supervisor-loss-check SUPERVISOR_LOSS_OUTPUT=outputs/evaluations/supervisor-loss-new
make supervisor-lifeline-test
```

实际使用 ProcessPairCapture(supervised=True)，合成 factory/reader 忽略 SIGTERM 并永久 sleep。确认读取进程进入阻塞后，只对监督进程发送 SIGKILL；主调用方仍存活。初始化和读取两个阶段各两次，共 4/4 通过：

- 外部 ps 核查读取进程均已不存在；判定发生在实验进程组兜底清理之前。
- 后端全部保留 supervisor_lost 和 workers_stopped=false，未误报成功回收。
- 未重试读取、未启动新会话、未进行机器人操作。

[实验记录](evaluations/supervisor-loss-v1.json)保存 PID、最终状态、完整后端清理报告及源码哈希。1 项新增连接断开测试与原监督、采集、恢复测试回归通过，共 29 项相关测试；Ruff 检查通过。

## 限制

这里的阻塞会释放 Python GIL。若读取进程同时被 SIGSTOP、原生代码长期持有 GIL或 OS 不可中断阻塞，而监督进程又死亡，本保护仍可能无法执行。没有消除双重故障，也没有实现更上层外部监督。

进程消失不等于相机资源已安全释放，内部状态保持未确认是刻意设计。实验是合成读取器、真实 OS 进程，未验证实机或断电恢复。该新增保护不改变之前两个数据库事务、证据时效或动作授权的边界。
