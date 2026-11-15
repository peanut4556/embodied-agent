# 跨进程检查点提案命令

2026-11-15。新增命令行入口，将上一轮 Python API 的预览与精确发布流程接成可保存 JSON 提案、稍后由新进程提交的操作流程。没有自动登记、自动审批或自动重试。

## 使用

已有源库和已登记仓库时，生成预览：

```sh
PYTHONPATH=src:. .venv/bin/python -m embodied_agent.checkpoint_cli preview \
  --database /path/evidence.sqlite --store /path/checkpoint.sqlite > proposal.json
```

必须检查退出码为 0 且 JSON status 为 proposed，再保存和审核提案。shell 重定向本身不是原子持久化协议，失败或中断可能留下错误报告或不完整文件；发布入口会拒绝这些文件，不自动修复。

明确决定发布后，传入原提案文件：

```sh
PYTHONPATH=src:. .venv/bin/python -m embodied_agent.checkpoint_cli publish \
  --database /path/evidence.sqlite --store /path/checkpoint.sqlite --proposal proposal.json
```

重新安装项目后也可使用 `embodied-checkpoint`。两种操作支持 `--max-records`（默认 10000）。提案文件最多读取 64 KiB，拒绝重复 JSON 键、错误结构、非成功预览及授权标志异常。请求校验在打开可写仓库之前完成；缺失仓库不会创建或重新登记。

| 退出码 | 含义 |
| --- | --- |
| 0 | 成功预览或成功发布，由 JSON status 区分 |
| 2 | 命令参数语法错误，包括 publish 缺失提案 |
| 5 | 请求、文件、存储或源库核验失败，查看 JSON 原因 |
| 6 | 仓库版本或源库候选内容变化，需要显式重新核查 |
| 7 | publication_unconfirmed，发布结果未确认；先重新读取仓库对账，不能直接认定失败并重试 |

预览只读；发布是明确写操作，可能触发 SQLite 热日志恢复。成功发布后同一提案再次使用返回版本冲突。提案可以跨进程传递，但不携带身份认证或审批签名；文件可编辑并不意味着编辑后的提案获得授权。应用仍须负责谁能提交、如何审批和保存审批记录。

## 验证

`make checkpoint-cli-test` 执行 6 项测试，所有命令都通过独立子进程运行，覆盖只读预览、发布与重复拒绝、合法新增导致旧提案失效、损坏/超大/重复字段提案、缺失仓库、参数错误和真实锁竞争的退出码 7。

另保存了一组实际三进程运行：预览返回 0，使用落盘提案发布返回 0，再次提交返回 6。[运行记录](evaluations/checkpoint-cli-v1.json)包含 JSON 输出、源文件及最终源库、仓库、提案文件哈希。69 项证据链、12 项仓库、11 项发布 API 与 6 项命令测试通过，共 98 项相关测试；Ruff 检查通过。

这一步推进的是可操作的跨进程工作流，不是新的分布式协议。实验输入仍为合成审计证据，使用虚拟证据时钟；没有真实相机、机器人、远端审批服务或断电可靠性结论。
