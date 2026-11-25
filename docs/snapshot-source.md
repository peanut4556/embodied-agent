# 快照来源声明与采集顺序检查

日期：2026-11-25。

## 新快照格式

export_snapshot 新增 source_id 和可选 capture_sequence。提供 source_id 时写入 schema 2，将来源与序号纳入完整快照摘要；省略时继续生成原 schema 1，不伪造默认身份。source_id 必须非空、无首尾空白、最多 128 字符；序号必须为正整数且不能是布尔值，单独提供序号而不提供来源会在写入前拒绝。

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.operation_admin snapshot \
  --checkpoint budget-state.json --output audit-002.json \
  --source-id robot-a-deployment-20261125 --capture-sequence 2
```

```python
summary = export_snapshot(
    checkpoint, output,
    source_id="robot-a-deployment-20261125", capture_sequence=2,
)
```

同一部署应使用稳定的来源声明；新的独立部署应分配不同标识。序号由调用方管理，本实现不创建跨进程序号注册表、不自动递增，也不会阻止单独导出两个重复序号文件；重复或倒序在比较时提示。序号不是设备时钟或可信时间戳。

## 比较前置检查

compare_snapshots 先校验两份文件，再判断来源与序号。新增 source 对象包含 before_id、after_id、relation、前后序号、order_relation、flags 和 override。

| 情况 | 默认行为 |
|---|---|
| 相同非空来源声明 | 可继续结构比较，relation=matched_declared |
| 来源声明不同 | source_identity_mismatch，暂停结构比较 |
| 任一侧 schema 1、无来源 | source_identity_unavailable，暂停结构比较 |
| 相同来源但后序号更小 | capture_order_reversed，暂停结构比较 |
| 相同来源和序号但文件摘要不同 | capture_sequence_reused，暂停结构比较 |
| 同一个快照与自己比较 | 不作为序号冲突，order_relation=same_capture |

序号递增时 order_relation=increasing_declared；缺少序号或来源不可比时为 unavailable。同源但未给序号仍允许结构比较，但不证明 before/after 顺序，操作者必须确认。暂停比较时 comparison_performed=false、counts=null、changes=[]、budget_flags=[]、needs_review=true，不输出可能误导的“预算倒退”判定。

## 兼容旧快照与显式继续

schema 1 仍可读取、校验、离线保存，不自动修改旧文件。行为变化仅在对比前置检查：旧快照默认比较会返回需要核对而非继续生成结构差异。已确认文件来源但希望比较旧数据，或确实需要比较不同声明的两份数据时，可显式继续：

```sh
PYTHONPATH=src .venv/bin/python -m embodied_agent.operation_admin compare-snapshots \
  --before old-audit.json --after audit-002.json --allow-source-override
```

API 对应 allow_source_override=True，只接受真正的布尔值。继续后 comparison_performed=true，但原 source.flags 保留，needs_review 仍为 true，CLI 仍退出 1，不能用开关消除来源警告。旧脚本若假设始终存在 counts 字典，需要先检查 comparison_performed。

验证与导出摘要增加 source_id、capture_sequence，旧快照对应 null。旧版本程序尚不识别 schema 2，应升级后读取，不能手工改 schema 或删除字段规避校验。

## 身份和真实性边界

source_id 是调用方声明，不是认证；攻击者可以复制来源标签、重排序号并重算摘要。工具仍输出 same_source_verified=false 和 physical_outcome_verified=false。相同声明只能排除明显的误选，不能证明两份文件确实来自同一设备，更不能验证物理任务结果。

当前没有持久绑定检查点身份、签名证书、可信时间服务或自动迁移协议。不要把每次随机生成不同 source_id 当成正确用法，也不要用同一通用标签覆盖所有部署。更换机器、复制目录或重新部署时应在外部运维记录中明确身份约定。

## 测试

新增 12 项测试覆盖 schema 2 及元数据往返、相同声明正常比较但不声称认证、不同声明提前阻断、schema 1 保留且不修改、混合版本显式继续、倒序、重复序号异内容、同快照比较、非法元数据不发布、重算摘要后的非法序号仍拒绝、真实 CLI 导出与 override 退出语义、布尔开关类型约束。

相关回归共 196 项通过。已有结构差异测试改为明确声明同一测试来源，以继续验证其原有回退规则；不是把来源检查关闭。历史实验和快照文件未被改写，原控制、预算、执行和检查点策略保持不变。

```sh
make snapshot-source-test
```
