# 审计快照的离线差异与回退线索

日期：2026-11-25。

## 比较入口

新增 compare_snapshots(before, after)，先对两份快照完整校验，再按回执键比较；不访问原检查点目录、不获取执行锁、不执行动作，也不修改输入文件。单份快照沿用 256 回执与文件大小限制，两份对比最多处理 512 个不同键。

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.operation_admin compare-snapshots \
  --before audit-before.json --after audit-after.json
```

```python
from embodied_agent.snapshot_diff import compare_snapshots
report = compare_snapshots("audit-before.json", "audit-after.json")
```

结果含两份快照摘要、变化分类计数、逐键变化及标记、预算变化与标记、needs_review、被忽略目录项数量。CLI 正常比较且没有规则标记时退出 0；比较成功但存在待核对项时退出 1；文件无效、摘要错误或读取失败退出 3；参数解析错误为 2。退出 0 不是安全认证或来源认证。

## 回执分类与核对规则

分类为 added、removed、changed、encoding_changed、uncertain，另统计 unchanged。有效回执结构相同而原文件字节摘要不同，归入 encoding_changed，不当作终态篡改。输入中任一侧源不可读则归 uncertain；即使两份坏源记录完全一样，也不能宣称正常未变化。

| 标记 | 含义 |
|---|---|
| receipt_missing | 之前存在的键不在后一快照纳入范围内，需检查删除、迁移或来源不一致 |
| request_rebound | 同键请求摘要变化，可能破坏原有操作 ID 绑定 |
| terminal_receipt_changed | complete 或 abandoned 的有效内容发生变化，包括回退到 pending |
| abandonment_predecessor_mismatch | pending 转 abandoned 时，终止记录的前驱摘要不对应前一回执 |
| unreadable_source | 一侧或新增记录无法有效读取，无法作可靠状态比较 |

正常 pending→complete 不标为异常。pending→abandoned 的前驱摘要匹配时也不标记。新增 complete 不能证明中间执行过程，但它本身不是不合法变化。工具不从键反推原操作 ID，不重新判定物理任务是否成功。

## 预算变化

有效预算分别报告前后 revision、新增退化动作和清除退化动作，并检查：

- budget_revision_decreased：预算版本倒退。
- calibration_history_not_extended：新历史不是旧历史的前缀扩展，包括截断或改写。
- budget_configuration_changed：回退值、裕量、最低样本数或策略发生变化。
- degradation_cleared_without_new_calibration：动作退化标记清除，但没有对应新增校准记录。
- estimate_changed_without_new_calibration：预算估计变化而没有对应新增校准记录。
- budget_unreadable：任一快照里的预算不可读，不猜测差异。

正常新增校准历史并恢复目标动作可以通过这些规则。标记只表示需结合上下文核对，不等于攻击或违规证明。例如上一快照恰好捕捉到执行前的保守预写隔离，下一快照在任务正常提交后清除这些标记，会出现“没有新校准的清除”；这可能完全符合既有执行流程。配置变化也可能是授权维护。工具不会因此自动隔离、回滚或恢复任何状态。

## 来源与时间边界

schema 1 没有来源字段；新增 schema 2 的来源声明与序号见 [来源检查](snapshot-source.md)，默认先检查可比性。两种格式都没有不可伪造的检查点身份或可信时间戳，比较函数也无法知道用户是否选了同一部署的正确先后顺序，所以始终输出 same_source_verified=false。必须由操作者核对部署、文件来源和 before/after 顺序。把两个不同项目的快照比较，可能产生大量没有意义的“回退”线索。

两份快照之间发生又恢复的变化不可见；忽略目录项只比较数量，不检查其中内容。哈希可被有写权限者重算，不是签名。physical_outcome_verified=false，历史成功和正常状态变化都不证明设备现在的物理状态。needs_review=false 仅表示在已校验且纳入范围的数据中没有触发本工具规则。

## 测试与兼容

新增 13 项测试覆盖：相同健康快照、正常完成与新增、回执消失、请求换绑、终态回退、终止前驱摘要匹配/失配、仅格式变化、相同坏源仍不确定、预算版本与历史倒退、合法重新校准与无记录清除、损坏快照拒绝、配置变化、真实 CLI 0/1/3 退出码和输入文件不变。与既有功能及共享解析器回归共 184 项通过。

为支持已验证内容读取，新增 read_snapshot 复用原结构校验；旧 verify-snapshot、导出、分页和操作执行行为保持不变。没有修改用户业务回执、历史实验结果或默认控制策略。

```sh
make snapshot-diff-test
```

接口兼容提示：来源信息不足或冲突时，现在默认返回 comparison_performed=false、counts=null；明确核对后可用 --allow-source-override 继续，但警告保留。
