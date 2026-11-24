# 有界操作巡检与回执分页

日期：2026-11-24。

## 巡检入口

新增 list_operations 和 operation_admin list，在同一检查点协作锁下扫描回执名称并读取选定页。每条保留 receipt_key、state；有效记录增加请求摘要与回执摘要，complete 记录增加历史 success/completed_steps。损坏、过大、结构非法或读取失败的记录显示 state=invalid 和异常类型，不阻断本页其他记录。

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.operation_admin list \
  --checkpoint budget-state.json --limit 20
```

```python
from embodied_agent.operation_inventory import list_operations
page = list_operations("budget-state.json", limit=20)
if page["has_more"]:
    next_page = list_operations("budget-state.json", limit=20, after=page["next_cursor"])
```

按 64 位十六进制回执键升序排列，游标是上一页最后一条键，下一页取严格大于游标的键。最后一页 next_cursor=null。空目录、尚未建立回执目录或游标超过末尾都返回空页；检查点无效仍会在 budget 中报错，不伪装成完全健康。

## 字段与统计范围

- items：当前页记录，包含损坏记录。
- page_counts：仅当前页的 pending/complete/abandoned/invalid 数量，不能当成全目录状态总数。
- matched_entries：本次扫描匹配规范文件名且为普通文件的数量，尚未验证全部内容。
- scanned_entries、ignored_entries：实际目录项数量以及非规范文件/目录/符号链接等未纳入回执页的数量。
- budget：当前检查点校验结果、版本、策略及退化动作集合；它与历史回执分开报告。
- has_errors：本页有 invalid 或预算无效。不表示其他未读取页也已检查无误。
- physical_outcome_verified=false：不证明物理结果。

CLI 退出 0 表示本页和预算没有发现错误；退出 1 表示仍成功输出本页，但本页有损坏记录或预算无效；退出 3 表示锁、扫描、输入等错误，stdout 为空，stderr 给出异常类型；参数解析错误为 2。外层 ok=true 表示清单生成成功，即使 has_errors=true 也不把已有数据丢弃。

## 资源限制

默认每页 20 条，合法范围 1..25；仅解析当前页文件，每文件使用既有 1 Mi 字符和嵌套限制。目录最多扫描 4096 项，所有项都计入上限，包括无关文件和临时文件。超出时抛 InventoryLimitExceeded，不返回截断清单。虽然有分页，每次调用仍须在这个有界目录内扫描并排序名称，不能称为无限规模索引。不会读取所有匹配文件内容来计算总体状态。

超过目录限制时需在保留重复请求证据的前提下设计后续分区或索引；本工具不提供删除/归档捷径。直接删除旧回执会丢失重复请求保护，不应为了通过扫描限制而盲删。单文件有界不等于硬实时，文件系统仍可能阻塞。

## 不知道原操作 ID 时查看详情

旧回执文件只保存 ID 的哈希键，不能从键还原原 ID。新增只读 inspect-key，接受清单里的键并返回完整已校验回执、摘要、预算状态及历史结果可用性：

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.operation_admin inspect-key \
  --checkpoint budget-state.json --receipt-key <清单中的64位键>
```

输入必须是严格 64 位小写十六进制，不接受路径或文件名。它不执行、不解封、不终止操作；终止仍使用原 operation-id 与前述检查摘要流程。实际部署应在上位机日志中保留 ID 与请求映射，清单不是原始任务日志的替代品。

## 一致性与只读边界

每次调用持有同一协作锁，避免与遵守协议的执行或终止操作同时读写。不同页之间不是同一个持久快照：新建键可能排序到已经翻过的范围，外部修改或删除也会改变后续页。需要完整一致的批次审计时应停止新增请求后扫描，不能声称游标有跨页事务保证。

清单排除符号链接以避免无意跟随非回执文件；路径仍要求可信本地目录，不能当成敌对文件系统安全沙箱。检查不修改回执与预算，但可能创建固定 .lock 侧文件。未知目录项被统计而非默默删除。检查不会验证 evidence_ref 指向的外部材料，也不能根据 complete 判定设备现在仍处于成功状态。

## 测试

新增 12 项测试覆盖混合状态与坏记录保留、三页无重复/遗漏（静态目录）、页内统计、空和缺失目录、非法游标与页大小、扫描超限拒绝、无关文件与符号链接统计、坏预算不隐藏回执、仅解析当前页、详情检查不改业务文件、真实 CLI 正常/损坏页退出码、锁冲突与键详情，以及越过末尾游标。相关回归共 155 项通过，源码检查通过。

```sh
make operation-inventory-test
```

新增 [持锁审计快照](operation-snapshot.md)：在更严格的数量与总读取限制内，将预算与全部匹配回执一次导出，并提供离线校验。分页本身仍不提供跨页快照保证。
