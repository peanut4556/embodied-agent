# 摘要一致之外的记录合同校验

2026-11-15。本轮发现双向索引审计仍可能放过的异常：两表摘要及会话字段一致，并不意味着正文符合原始 receipt 合同，或者记录的接收时间在原始有效期内。

## 实现

账本读取 `_lookup` 在 SHA-256 核对后解析正文，复用写入时的 receipt 校验，并额外验证：

- 正文必须是当前写入器生成的规范 JSON 对象，包含 consumed 状态、confirmed_recent_evidence 状态、合法编号、会话和时间字段；robot_action_authorized 必须为 false。
- 正文编号和会话必须与查询编号、数据库会话字段一致。
- 数据库 accepted_host_time 必须有限，且满足 consumed_host_time ≤ accepted_host_time < expires_host_time。

不满足返回 `invalid_record`，与摘要损坏的 `integrity_error` 分开。共享读取路径使按编号对账、索引发现和诊断命令同时获得校验；审计遇到 invalid_record 会返回 blocked（5）。已有异常记录不能通过再次提交相同正文获取 duplicate 成功回执。错误形状的写入参数统一返回 ValueError。

这是历史接收合同检查，不把历史记录拿当前时间判断过期；合法旧记录仍可核对，也不会被续期。正文内部的完整相机算法未重算，仅复核现有 receipt 接口合同，不声称它证明真实采集成功。

## 可复现反例

```sh
make evidence-semantics-check EVIDENCE_SEMANTICS_OUTPUT=outputs/evaluations/evidence-semantics-new
```

每个案例先登记一条正常记录的检查点，然后追加第二条记录并注入异常，避免旧检查点摘要提前拦截：

| 注入 | 摘要和两表绑定 | 新诊断 |
| --- | --- | --- |
| 接收时间改到原始有效期之外 | 一致 | invalid_record，blocked |
| 正文会话修改并同步更新两表摘要 | 一致 | invalid_record，blocked |
| 正文授权字段改为 true 并同步更新两表摘要 | 一致 | invalid_record，blocked |

3/3 独立 CLI 运行拒绝异常，均确认检查点历史仍匹配，文件读取前后哈希不变。[实验记录](evaluations/evidence-semantics-v1.json)保存全部结果和代码、输入文件哈希。新增 4 项参数化及回归测试，66 项证据链测试和 12 项仓库测试通过，共 78 项相关测试；Ruff 检查通过。

## 边界

这消除的是“字节摘要与索引一致，但内容违反自身合同”的漏查。攻击者若连正文和相关字段都一致伪造为合法值，仍不保证检出；它不是签名认证或真实性证明。实验使用合成 receipt，没有实机、断电或实时性能结论。

读取会额外解析和规范化单条 JSON，增加计算成本；历史校验仍适合离线诊断。原写入器始终保存规范 JSON，正常记录兼容；外部手工写入的非规范 JSON 即使语义相似也被拒绝，不自动改写迁移。版本升级若调整序列化合同，需要显式迁移方案。
