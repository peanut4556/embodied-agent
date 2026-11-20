# 基础运行链路的逐步观测确认

日期：2026-11-20。

本轮转回自然语言到动作的基础 AgentRuntime。原实现对每次 execute 调用直接累计完成步数，所有调用不抛异常就报告成功；因此机器人静默未抓到或未放下，也可能得到 task completed。

现在每步执行后重新 observe，并按结构化动作校验：locate 要求对象存在；pick 要求 gripper_holding 等于目标且对象位置为 gripper；place/verify 要求对象位于显式 destination 且夹爪为空。未满足时返回 success=false，completed_steps 只统计已核验步骤，并停止执行后续动作。stop 执行后返回 task stopped，不再宣称任务成功。

策略收到计划步骤的深拷贝；返回动作名称及计划中已有参数必须一致，偏离时在执行前返回失败。额外参数仍接受原 SafetyGate 校验。这里不解析自然语言 success_condition，不尝试自动重试，不发送额外补偿动作；失败只是结束主循环，不能替代硬件急停。

兼容性变化：仅报告命令已发送、未及时更新符号观测的异步适配器，现在会得到未确认结果，需要调用方另行定义等待与观测合同。verify 没有 destination 时不能证明放置目标，因此拒绝确认。接口异常仍按原行为向上抛出。自定义 SafetyGate 允许的未知动作没有确认规则，也不会自动判成功。默认 mock 四步路径仍成功。

新增 5 项测试：静默 pick 失败后不执行 place；静默 place 失败后不执行 verify；策略替换目的地不执行；无可观测目标不抓取；stop 不报告成功。加原有运行及安全校验共 9 项测试通过，Ruff 检查通过。实际执行默认中文 CLI 命令返回 success=True、completed_steps=4。

```sh
make runtime-confirmation-test
PYTHONPATH=src .venv/bin/python -m embodied_agent '把桌上的红色积木放进盒子'
```

这是基础入口行为的实质修复，没有宣称泛化成功率提升。观测来自 Robot 适配器，可能陈旧或错误；目前没有时间戳新鲜度、独立视觉证明或接入前面高精度证据链。MuJoCo/控制台的独立执行器并未由本轮改变，也没有实机实验。
