# 绑定预览内容的检查点发布

2026-11-15。严格发布保证当前内容符合合同，但如果操作方先审核、稍后发布，期间新增的合法数据也可能被一起纳入。本轮新增只读预览与精确提案发布，要求发布的内容仍等于调用方提供的已审阅提案。

```python
preview = preview_publication(database_path, store_path)
if preview['status'] == 'proposed':
    proposal = preview['proposal']  # 由应用展示、保存并组织审核
    # 审核结束后，显式提交原提案：
    with CheckpointStore(store_path) as store:
        result = store.publish_reviewed(database_path, proposal)
```

提案包含仓库 generation 和完整候选检查点（数据库身份、两表计数、高水位及摘要），不含证据正文。预览先核对旧历史并执行严格完整性门槛，不写入源库或仓库。

发布时仍持有仓库写事务，先比较 generation，再在同一源库读快照中验证旧历史、检查完整证据、生成候选并与提案精确比较。即使变化只是合法追加，也返回 `source_rejected` / `proposal_changed`，保留旧 head。需要调用方显式重新预览和审核，代码不会自动替换提案。重复发布则因 generation 已变而拒绝。

## 验证

```sh
make reviewed-publication-check REVIEWED_PUBLICATION_OUTPUT=outputs/evaluations/reviewed-publication-new
make reviewed-publication-test
```

持久化实验：基线 1 条记录，追加至 2 条后生成提案，再合法追加至 3 条。旧提案发布被拒绝、head 未变化；重新预览的 3 条提案发布成功；再次使用该提案被版本冲突拒绝。最终存储的检查点与新提案精确相同。[运行记录](evaluations/reviewed-publication-v1.json)保存两份提案、三个发布结果及源代码、文件哈希。

新增 5 项测试覆盖只读预览、精确发布、合法新增导致提案失效、重新预览、重复请求、缺失仓库、非法提案和检查数量上限。连同原 87 项相关测试，共 92 项通过，Ruff 检查通过。数据为合成 receipt、虚拟证据时钟及真实 SQLite 文件，未进行实机验证。

## 边界

提案是可信调用方持有的一致性令牌，不是签名、审批记录或授权凭证；本轮没有新增审批 UI、身份验证或审批人审计日志。能更改提案的调用方仍可提交其他有效提案。

约束的是被捕获和发布的快照内容，不是保证源库在仓库提交时毫无变化；两个库没有分布式事务。源库在该读快照之后追加的数据仍可能未被新检查点覆盖，需要再次诊断。完全相同的内容经历修改再恢复无法据此判断中间过程。旧 advance / advance_verified 接口仍保留，需要绑定审核内容时应显式选择新入口。

后续操作入口：[跨进程检查点提案命令](checkpoint-cli.md)提供 preview / publish 子命令及文件校验、分级退出码。
