# 第二路见证确认接口

日期：2026-11-22。

本轮针对持续错误主观测的边界，给 AgentRuntime 增加可选 confirmation_witness(action) 回调。回调收到动作的深拷贝，不接收主观测，返回单独 Observation；默认 None 保持旧行为。启用见证必须同时配置 max_observation_age_s。max_witness_skew_s 默认 20 ms，允许有限值 0–100 ms。

每轮先读主观测，再调用见证；见证返回后重新检查截止时间与取消，同时复核主观测此时仍未过龄。两路必须各自满足已有符号动作后置条件，取样时间不早于确认起点、不晚于当前时钟、未过龄且时间差不超过门槛。结果冲突或未对齐不能确认，受原有轮询时限和读取次数约束；非法见证或时间戳立即拒绝。

多样本模式中，两路时间都推进才增加样本数，任一路缓存不会凑数。见证时间倒退拒绝；任一路间隔过长重新开始。取样跨度保守计算：本段初始两路中较晚时间，到当前两路中较早时间。报告新增 witness_reads、witness_disagreements、witness_unaligned；disagreements 表示两路符号条件判断不同，不是物理真值。

```python
runtime = AgentRuntime(
    planner, policy, robot,
    confirmation_timeout_s=0.5,
    max_observation_age_s=0.1,
    confirmation_samples=3,
    confirmation_witness=second_sensor.observe_for_action,
    max_witness_skew_s=0.02,
)
```

第二传感器适配器由调用方实现，没有提供把主观测复制过去的默认“见证”，CLI 也未假装连接硬件。回调按当前动作获取同一对象和目的地的实际观测，时间戳需映射到相同单调时钟域。

新增 5 项测试覆盖持续主源正面且见证负面的冲突拒绝、双方一致的多样本确认、缓存见证不能累计、陈旧/未来/非法见证拒绝及未配置新鲜度拒绝。与既有主运行测试共 40 项通过，Ruff 检查通过。

```sh
make runtime-confirmation-test
```

本轮是接口与状态逻辑接入，不是物理独立传感器验证。两个来源若共享错误或回调伪造正确状态，仍会通过；未估计独立性或联合错误率。串行调用可能增加延迟或降低可用性，阻塞回调无法由同步循环强制中断。回调异常按既有方式向上传递，没有自动采集重试或动作重放。此前合成压力测试未在该双源配置上重新运行，不能宣称其 30 个误接受已全部解决。未做相机、MuJoCo 或实机实验。
