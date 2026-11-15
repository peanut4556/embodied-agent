# 调用方死亡后的采集工作进程自行退出

2026-11-15。本轮新增可选 owner_watchdog，处理上一轮尚未覆盖的生命周期问题：调用方未执行 cancel/reap 就突然退出，而采集工作进程卡在初始化或读取中。

## 设计与接入

```python
backend = ProcessPairCapture(reader_factory, owner_watchdog=True)
# 或在恢复管理器中显式开启：
manager = CaptureRecovery(position_error_m=(0.001,)*3, owner_watchdog=True)
```

每个工作进程有独立单向存活连接，唯一写端由调用方持有，工作进程只获得读端。spawn 启动后在 factory 执行前建立守护线程，阻塞等待该连接；调用方死亡或主动关闭写端会产生 EOF，线程直接 os._exit(86)。没有轮询心跳，也不依赖主读取线程返回或响应 SIGTERM。

正常 cancel 同时关闭存活连接，并保留既有 terminate/kill/reap 路径。状态报告增加 owner_watchdog_enabled。默认关闭，旧接口和测试的终止行为不变；通过 CaptureRecovery 开启时每轮新后端都会使用新连接。

## 实验

```sh
make owner-watchdog-check OWNER_WATCHDOG_OUTPUT=outputs/evaluations/owner-watchdog-new
make owner-watchdog-test
```

独立调用方启动两个工作进程，合成 factory/reader 忽略 SIGTERM 并永久 sleep。工作进程写入“已进入阻塞”标记后，调用方直接 os._exit(93)，刻意不执行 cancel、reap 或 finally。监督程序在调用方退出后通过 ps 查询工作进程状态，验证无需调用方主动清理即可退出。

两类场景各两次，共 4/4 通过，8 个 PID 在核查时均已不存在：

| 场景 | 调用方退出 | 工作进程观察结果 |
| --- | --- | --- |
| 读取器阻塞 ×2 | 93 | 全部消失 |
| factory 初始化阻塞 ×2 | 93 | 全部消失 |

记录中的 observation_elapsed_s 是调用方退出后执行观测的耗时，不是精确的工作进程退出延迟，不能作为实时期限承诺。若实验失败，监督程序只清理本次专属进程组；本轮通过条件均在该兜底清理之前核对。

[最终实验记录](evaluations/owner-watchdog-v2.json)包含源码哈希、退出码、PID 及观察状态。首轮沙箱禁止 ps，无法完成退出核验；保留该目录，不计作通过。在获得本地进程检查权限后执行 v2 得到上述结果。

新增 4 项测试验证存活连接断开时退出码 86（读取卡住/初始化卡住）、恢复管理器接入与正常关闭、参数校验；原进程后端 9 项和恢复管理器 9 项测试回归通过，共 22 项。Ruff 检查通过。

## 边界

本轮推进点是把“调用方死后必须由外部清理”扩展为工作进程可自行察觉失去所有者，且在合成阻塞读取中真实验证。不同于上一轮退出前先回收资源，本轮调用方确实没有清理工作进程。

监测线程依赖 Python 线程获得运行机会。原生驱动若长期持有 GIL、整个进程被暂停、OS 不可中断等待或另有进程泄漏存活连接写端，不能保证及时退出；未验证这些情况。sleep 型阻塞会释放 GIL，不可宣传为解决任意驱动死锁。

os._exit 跳过清理回调，证明进程退出不证明真实相机或机械设备已恢复安全状态。没有实机验证、硬件清理保证、自动重启或动作授权。该选项为可选机制，尚未替代外部监督及设备恢复策略。
