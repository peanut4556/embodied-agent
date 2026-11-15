# 严格检查点发布准入

2026-11-15。此前诊断能报告证据异常，但普通检查点推进只要求旧历史未改变，可能将异常追加内容记录为新基线。本轮新增可选严格入口 `CheckpointStore.advance_verified`，在同一源库快照中验证待发布内容，再生成检查点。

```python
with CheckpointStore(store_path) as store:
    result = store.advance_verified(
        evidence_path, expected_generation=observed_generation, max_records=10000,
    )
```

它沿用仓库版本比较和事务提交；读取源库时先验证旧检查点，再核对全部索引对应有效已落账记录，并反向拒绝无索引记录。之后在同一读事务内生成新检查点。未落账、无索引、摘要/正文合同错误均返回 `source_rejected`，内部原因是 `evidence_rejected`。任一表行数超过上限则为 `validation_incomplete`，不会部分批准。失败保留旧 head，不自动修复或重放。

`checkpoint_evidence(..., require_complete=True, max_records=...)` 提供同样的底层可选校验。旧的 `advance`、首次登记接口保持原合同，仍可保存存在待提交项的历史基线；需要“全部完成且一致”的业务应显式选择新入口，不能认为所有发布接口都已自动变严。

## 实验与回归

```sh
make verified-publication-check VERIFIED_PUBLICATION_OUTPUT=outputs/evaluations/verified-publication-new
```

6/6 个持久化场景符合预期：正常追加成功推进到 generation=2；无索引、未落账、非法接收时间、正文损坏及超过上限五种情况全部拒绝，仓库 head 与调用前一致。源数据库每次发布前后哈希不变。[实验记录](evaluations/verified-publication-v1.json)保存结果及源代码、输入文件哈希。使用合成 receipt、虚拟证据时钟和真实 SQLite 文件，未接入相机或真实设备。

新增 6 项测试（含多种故障子案例）及原 81 项相关测试通过，共 87 项。WAL 并发测试在校验完成后、生成检查点前删除一条源记录，确认新检查点仍包含校验时的两条记录，随后诊断检出删除。这证明校验与生成不混用两个源库快照；不意味着源库在检查点提交时仍保持不变。

## 边界

本轮闭合的是“诊断异常也能被普通发布纳入基线”的操作缺口。发布仓库事务与源库读事务仍不构成分布式原子提交；源库后续变化要靠再次诊断检测。合法一致的伪造内容仍无法凭此认证。

行数上限限制正文校验数量，不限制 COUNT、旧历史摘要扫描和查询总耗时；这是离线操作。严格发布可能因正在进行的合法提交而暂时拒绝，调用方应先核对状态，再决定是否显式重试。没有自动重试、实机动作授权、断电保证或新增远端见证。
