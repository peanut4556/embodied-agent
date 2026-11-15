# 提案发布回执丢失后的事务对账

2026-11-15。本轮解决一个此前版本冲突无法解释的问题：发布成功但回执丢失时，重交提案会被拒绝，却不能直接确认原提交是否成功。新增与检查点 head 同事务保存的提案发布记录，以及只读历史查询。

## 合同与入口

`publish_reviewed` 成功时，在仓库事务内同时写入 `checkpoint_publications_v1`。记录包含规范提案 JSON 的 SHA-256 身份、原提案、提交后的 generation 和覆盖正文与 generation 的校验和。索引以提案内容为身份；同样内容的不同调用方无法据此区分，不包含操作人身份或网络请求编号。

```python
result = reconcile_publication(store_path, original_proposal)
```

也可以读取上一轮 CLI 保存的完整 preview 文档：

```sh
PYTHONPATH=src:. .venv/bin/python -m embodied_agent.checkpoint_cli reconcile \
  --database /path/evidence.sqlite --store /path/checkpoint.sqlite --proposal proposal.json
```

该命令只查询仓库；`--database` 保留公共命令参数合同和文件分离校验，不扫描源库。需要检查当前源库时另行运行 evidence_audit。

- `publication_recorded` / 退出码 0：查到该提案的历史提交，返回提交 generation、当前 generation 和 historical_only=true。
- `no_publication_record` / 退出码 8：没有该提案的发布记录，不能因此推断它从未在别处、旧版本或已回退的仓库中成功。
- 仓库不可读、损坏、head 与回执不一致等返回错误状态，CLI 为 5；查询不创建文件、不自动修复、不重放。

普通 enroll / advance / advance_verified 不生成提案回执，旧版本历史也不会自动补写。重复 publish_reviewed 仍因版本冲突拒绝，调用方应显式 reconcile，而不是靠重试获取成功。历史发布之后即使仓库继续推进，仍能查询旧回执。head 低于回执版本或同版本内容不符时报告 history_mismatch。

## 故障验证

```sh
make publication-receipt-check PUBLICATION_RECEIPT_OUTPUT=outputs/evaluations/publication-receipt-new
```

独立 spawn 发布进程，真实 SQLite、真实 os._exit，合成 receipt 与虚拟证据时钟：

| 每种两次 | 退出码 | 重启 head | 按提案查询 |
| --- | --- | --- | --- |
| 正常提交 | 0 | 2 | publication_recorded |
| COMMIT 前退出 | 91 | 1 | no_publication_record |
| COMMIT 后、回执前退出 | 92 | 2 | publication_recorded |

6/6 符合预期，所有发布进程回收；恢复只打开仓库让 SQLite 处理热日志，再只读查询，没有重交提案。[运行记录](evaluations/publication-receipt-v1.json)保存结果与代码、源库、仓库和提案文件哈希。新增 4 项测试及原 98 项相关测试通过，共 102 项；现有 CLI 测试额外验证 reconcile 子命令。Ruff 检查通过。

## 边界

这是本地发布事务的可追溯结果，不是机器人动作成功或当前源库一致性保证。仓库整体回退或历史被删除仍可能丢失回执，未新增更外部的见证。校验和不是签名，恶意同步改写不在认证范围内。发布记录会累计增长，本轮未加入清理/归档策略。故障覆盖进程退出，不证明断电持久化可靠性。
