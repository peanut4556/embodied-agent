# 主运行链路的观测新鲜度合同

日期：2026-11-21。

AgentRuntime 新增可选 max_observation_age_s，None 保持旧行为，启用时必须为有限正数且不超过 5 秒。确认观测使用 metadata.observed_at_monotonic，与运行时 clock 属于同一单调时钟域。

动作 execute 返回后，确认循环记录 started。仅当 started <= observed_at_monotonic <= 当前时间，且年龄不超过配置阈值，才允许已有符号后置条件判定成功。旧或过龄样本不确认，可在既有等待时限内继续轮询；缺失、布尔、非数值、NaN/Infinity 或未来时间戳立即返回 invalid_observation_timestamp。不重复 execute，不改写传入观测时间。默认零等待仍可核验当前样本。

时间边界是确认循环开始，不是发送动作时刻，故 execute 内已采到的状态即使正确也可能被保守拒绝，需要后续取样。严格模式仅校验动作后的确认观测，不对规划/策略的前置观测作新鲜度保证。

新增 TimestampedMockRobot 在每次同步读取 mock 状态后标记 time.monotonic；它读取的是内存状态，不是相机曝光。原 MockRobot 保留不变。CLI 使用 --max-observation-age 显式选择带时间戳 mock 和新鲜度门槛，可与 --confirmation-timeout 同时使用。

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent '把桌上的红色积木放进盒子' --max-observation-age 0.1
make runtime-confirmation-test
```

新增五项测试覆盖动作前旧成功状态不通过、旧后新样本成功且动作只调用一次、非法/未来戳拒绝、立即模式当前样本成功、非法年龄配置拒绝。与既有主运行回归合计 20 项测试通过，Ruff 检查通过。实际严格模式 mock 命令成功完成四步。

适配器必须报告真实取样时刻，不能对缓存数据重盖时间戳；时钟同步和跨设备映射需另行实现。这个合同减少陈旧状态误判，但不证明观测真实性、任务因果关系或硬件安全，也不强制终止阻塞 observe。没有独立相机证据、实机实验或统计成功率提升结论。

更正（2026-11-21）：后续检查发现本轮 CLI 参数虽已添加，但未传给运行时，因此上文严格模式 CLI 成功不能作为该版本启用新鲜度的证据；API 测试结果仍有效。该遗漏已在连续多样本确认扩展中修复，并增加 CLI 接线测试及实际严格命令验证，详见 [连续确认](runtime-multisample.md)。
