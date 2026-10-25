# 释放确认结果语义 v2

2026-10-25。修正上一轮发现的状态歧义：视觉确认区域比实际托盘更保守，区域外观测不能直接证明物理落点在托盘外。本轮改变结果命名与解释，不修改区域、速度门槛、采样或等待时间。

## 接口

ReleaseConfirmation.observe() 仍返回状态字符串；新增 result() 返回独立快照，包含 schema_version=2、status 和 reason。调用方不能通过修改快照改变监控状态。终止状态和原因保持不变，后续观测不会重启验证。

| 状态 | 原因 | 含义 |
|---|---|---|
| pending | collecting_evidence | 正在收集证据 |
| confirmed_in_tray | stable_visual_evidence | 已满足连续视觉确认条件，不等同于完整抓放成功 |
| outside_confirmation_region | observed_outside_confirmation_region | 超时时当前可见位置不在保守确认区域；物理落点仍可能在托盘内 |
| unconfirmed | missing_observation | 超时当次没有有效观测，不表示此前始终不可见 |
| unconfirmed | insufficient_stable_evidence | 超时当次位置在区域内，但连续稳定证据不足 |

这是实验接口的兼容性变更：旧 outside_tray 字符串不再产生，消费者需改为 outside_confirmation_region，并避免将其映射为确定的物理失败。旧报告保留原样；新报告显式标记 confirmation_schema_version=2。观测分支的逐次记录也携带 reason，完整回放报告增加 confirmation_result；未启动时为 null。

## 回归结果

重跑此前相同的 22 组冻结视角压力案例，并逐条比较旧报告：观测时间、视觉位置、物理位置和姿态、检测错误、观测次数完全一致；仅状态名称及新增原因字段变化。

- 6 例 confirmed_in_tray，均为 stable_visual_evidence。
- 10 例 outside_confirmation_region，不再声称物体必然在托盘外。
- 6 例 unconfirmed，其中 4 例在超时当次缺失观测，2 例稳定证据不足。

仍有 4 个实际在托盘内的案例未确认；本轮没有提高确认率，也没有消除这些视觉限制。仍未观察到误确认。典型边界反例 roll-0.69-30 现在准确表达为“确认区域外”，其物理真值仍在托盘内。

新增测试覆盖边缘和高空位置、缺失与不稳定证据的原因区分、结果快照、终止原因锁定，并补查观测分支的原因透传。共 99 项测试通过：58 核心、31 接触、10 释放相关。Ruff 与 diff 检查通过，物理运行前后源文件哈希一致。图形环境仍有 ARB_clip_control 深度精度提示。

机器报告：[22 组语义 v2 回归](evaluations/release-semantics-v2.json)。本轮没有重跑完整抓放物理回放，观测分支字段透传由测试验证；既有控制逻辑未改。

## 复现

```sh
make release-semantics-evaluate RELEASE_SEMANTICS_OUTPUT=outputs/evaluations/release-semantics-new
```

需要 MuJoCo 图形连接，输出目录必须不存在。下一步应针对未确认样本验证观测质量和时间预算，用独立对照决定是否调整参数，而不是把未知结果显示为失败。
