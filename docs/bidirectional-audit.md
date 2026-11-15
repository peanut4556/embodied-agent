# 双向证据一致性审计

2026-11-15。本轮修复只读诊断的一项实际漏查：检查点生成后新增的证据，如果索引丢失，单向“索引→证据”扫描仍可能看起来正常。新增反向“证据→索引”检查，并纳入原 CLI 的退出判断。

两方向共用原有源库读事务。反向逐批读取证据行，通过 receipt_id 左连接索引，报告 `indexed`、`unindexed` 和 `binding_conflict` 数量。无索引记录或会话/摘要绑定不符均返回 blocked（退出码 5）；不自动补写索引，也不推断这是篡改还是旧写入器混用。

`--max-records` 现在分别限制两方向的扫描行数，每批至多 100 行。输出保留原 `examined` 和 `counts`，增加 `records_examined`、`record_counts`、`index_scan_complete` 和 `record_scan_complete`。只有两方向都完成，`scan_complete` 才为真。尚未查到异常但任一方向未完成时返回 incomplete（4）；已发现异常则优先 blocked。原先 verified 的含义收紧为历史核验通过、双向扫描均完成且记录相互对应。

## 可复现反例

```sh
make bidirectional-audit-check BIDIRECTIONAL_AUDIT_OUTPUT=outputs/evaluations/bidirectional-audit-new
```

实验先写入一条证据并登记检查点，再新增第二条，随后只删除第二条索引。旧的发现接口仍返回 scanned，且只看到第一条已落账记录。新 CLI 完整扫描返回 blocked，报告一个无索引记录；上限为 1 时，正向已查完而反向未查完，返回 incomplete，避免错误通过。两次 CLI 运行前后数据库和仓库哈希相同。

[保存的运行记录](evaluations/bidirectional-audit-v1.json)包含前向结果、CLI 输出、源代码及输入文件哈希。新增 4 项测试覆盖尾部索引删除、普通账本写入器绕过索引、反向扫描限制及绑定冲突；全部 62 项相关证据测试通过，Ruff 检查通过。

突破点是补齐此前无法从索引发现的账本内容，实际拒绝了一类旧扫描可遗漏的异常。数据仍为合成审计输入，未验证实机。历史摘要校验仍可能扫描大量数据；行数限制不等于总耗时限制。双方内容被一致改写且不受旧检查点覆盖时仍不保证检出；不提供身份认证、动作许可或真实数据完整性证明。
