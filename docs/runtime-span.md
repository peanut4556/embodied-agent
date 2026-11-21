# 带时间跨度的动作确认

日期：2026-11-21。

AgentRuntime 新增 confirmation_span_s（默认 0）和 max_confirmation_gap_s（默认 None）。跨度非零时必须显式配置最大间隔，且样本数至少 2；沿用多样本模式对新鲜度、正超时及足够读取次数的要求。跨度必须有限、非负且不超过确认超时，间隔必须有限正数且不超过 5 秒。

确认同时要求累计有效样本数达到 confirmation_samples，最后一个与本段第一个有效正面样本的取样时间差至少为 confirmation_span_s。新正面样本与上一返回样本时间差超过 max_confirmation_gap_s 时，丢弃旧段，从该样本重新开始。反例或不新鲜样本也清零；重复时间戳不增加样本数，也不延长取样跨度。时钟倒退、时间戳倒退、取消和总时限仍按原合同处理，动作没有重执行。

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent '把桌上的红色积木放进盒子' --confirmation-timeout .5 --max-observation-age .1 --confirmation-samples 3 --confirmation-span .05 --max-confirmation-gap .03
```

实际命令在同步 mock 上完成四步。新增 6 项测试覆盖密集短突发拒绝、覆盖跨度后通过、长间隔重置、反例及重复帧不能凑跨度、非法配置、CLI 参数接线。与既有运行回归共 32 项测试通过，Ruff 检查通过。

```sh
make runtime-confirmation-test
```

所有新增条件默认关闭，旧行为保持。参数组合合法不保证一定可达，例如读取次数较少或轮询间隔大于允许间隔时，可能一直无法确认，最终受原有超时及读取上限停止。跨度在有效观测时间戳上计算，不用轮询耗时代替取样跨度。

这些条件覆盖有限的离散样本，不证明相邻样本之间状态没有变化，不构成物理稳定性、统计独立性或硬件安全保证。适配器仍需真实报告取样时刻。不同动作目前共享同一确认配置，本轮未做动作类别自适应、MuJoCo 或实机性能实验。
