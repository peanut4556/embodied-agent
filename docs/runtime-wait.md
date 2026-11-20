# 有时限的动作确认等待

日期：2026-11-20。

基础 AgentRuntime 新增可选 confirmation_timeout_s（0–5 秒）、confirmation_poll_s（1 ms–1 秒）、max_confirmation_reads（1–1000，默认 128）。默认超时为 0，保留单次立即核验。正超时时，每步 execute 仍只调用一次，后续只 observe，满足原符号后置条件才能进入下一步。

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent '把桌上的红色积木放进盒子' --confirmation-timeout 0.5
```

截止时间从 execute 返回后、确认循环开始时计算。单调时钟必须有限且不倒退；正超时时 observe 返回已超过期限，即使结果符合条件也拒绝。无正面结果时按轮询间隔等待，受时间及读取次数双重约束。run 新增可选 stop_requested 回调，启动前、动作执行前及确认等待中检查；取消返回失败，不发送额外动作，不重试。completed_steps 只包含此前已确认步骤。

新增 6 项测试使用可控时钟与延迟符号适配器：异步抓放最终通过且每个动作仅调用一次、超时不继续放置、等待期间取消、读取上限与非法时钟、迟到正面观测拒绝、非法配置拒绝。与原基础运行回归共 15 项测试通过。默认及显式 0.5 秒模式的正常 mock CLI 均完成四步；Ruff 检查通过。

```sh
make runtime-confirmation-test
```

这是异步状态确认能力，不是动作重试或硬件恢复。时限不包含 execute 耗时，也不能抢占阻塞中的 observe、sleep 或 stop_requested；取消检查是协作式，不是独立急停。默认 0 模式仍接受单次 observe 返回的结果，没有耗时硬限制。停止轮询不会撤销已发送的动作，真实设备需要独立停止控制层。

轮询只确认 Robot 返回的符号状态，不检查图像、时间戳新鲜度或独立观测序列号；陈旧但符合条件的观测仍可能通过。没有改变 MuJoCo 或控制台的独立执行器，没有实机成功率实验。后续应由适配器定义观测新鲜度合同，而不是简单增加轮询次数。
