# 预算与动作暂停状态的检查点

日期：2026-11-24。

## 保存与恢复

新增 budget_checkpoint.save_budget/load_budget，保存估计值、退化动作集合、策略、裕量、最低样本数、版本和完整重新校准历史，包括验证批次。恢复到新进程后，同类动作仍被 quarantine 阻止，旧批次不能再次使用，其他动作可继续按照预算运行。文件缺失或损坏会抛异常，不自动创建一个没有暂停状态的新预算。

```python
from embodied_agent.budget_checkpoint import load_budget, save_budget

budget = load_budget("budget-state.json")
result = runtime.run(instruction, task_timeout_s=2, action_budget=budget)
save_budget(budget, "budget-state.json")
```

首次初始化应由调用方明确创建 ActionBudget 后保存；不要把所有加载错误捕获后自动回退为空状态。重新校准时，先校验并创建候选，再保存候选成功后采用：

```python
replacement = budget.validated_recalibrated(
    "verify", calibration, batch_id="cal-003",
    validation_samples=holdout, validation_batch_id="val-003",
)
save_budget(replacement, "budget-state.json")
budget = replacement
```

这些是显式保存 API，不是自动事务系统。任务中发生退化到保存完成之间仍存在崩溃窗口；进程在保存前终止可能留下旧检查点。不能声称任何崩溃都不会丢状态。保存失败应报告并停止新的任务派发，由应用决定处置，不应继续假定新状态已持久化。当前运行 CLI 未自动接入检查点。

## 严格校验

文件为 schema 1 JSON 外层封装与规范化状态 SHA-256。拒绝未知/缺少字段、重复 JSON 键、NaN/Infinity、非法数字、超过 64 层嵌套和超过 1,048,576 字符的输入。输出采用 ASCII 转义，同样限制长度。集合上限为 128 个估计动作、128 个退化动作及 1000 条校准历史。

恢复检查版本与历史长度相符、历史版本连续、动作名与批次格式正确、批次在校准和验证角色间无重复、样本数达到配置要求、摘要格式正确、验证最大值不超预算，以及每个动作最新历史预算与当前估计一致。保存前也执行这些校验，内存中的非法状态不能覆盖旧文件。退化集合可包含没有校准估计的动作，因为未知动作使用回退预算，也可能发生故障。

摘要仅检测非恶意损坏，不是签名。能编辑文件并重新计算摘要的人可以伪造状态；旧文件回滚也无法检测。原始校准样本未存储于检查点，无法用摘要重新证明校准数据真实性。

## 文件替换与失败语义

先在目标同目录创建临时文件，写入、flush、文件 fsync 后使用 os.replace 原子替换；替换失败时清理临时文件，旧目标保持完整。保存目录必须已存在。不实现目录 fsync、电源故障持久性保证、并发写锁或版本比较交换；仅适用于可信路径、单写入者和不并发修改的预算对象。不能据此声称防恶意篡改或跨文件事务。

## 只读检查命令

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.budget_checkpoint budget-state.json
```

校验成功输出 JSON 摘要（策略、版本、退化动作、当前所需预算、历史数量），退出 0；失败向 stderr 输出 JSON 原因，退出 2，stdout 为空。命令不修改文件、不恢复运行、不执行机械动作。它适用于排查，不是一个解封入口。

## 验证

新增 8 项测试：完整状态/历史往返与任务拒绝、实际新 Python 进程加载、损坏及缺失文件不降级、重新计算摘要后的结构不一致仍拒绝、歧义/非有限/超大/深嵌套 JSON 拒绝、模拟 replace 失败保留旧文件并清理临时文件、非法内存状态不覆盖、只读命令成功与失败语义。往返测试还覆盖恢复后批次防复用，以及单动作重新校准后再次保存加载。

相关回归共 103 项通过，Ruff 检查通过。没有改动实验冻结结果或默认策略，也没有实机、断电或多进程写入验证。

```sh
make budget-checkpoint-test
```
