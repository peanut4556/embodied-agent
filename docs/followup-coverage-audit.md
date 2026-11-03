# 观测覆盖率的独立离线复核

2026-11-02。覆盖诊断新增逐条依据，离线工具通过独立区间并集重新计算各项指标，避免仅凭在线累计值判断覆盖情况。区域告警和采集生命周期规则不变。

## 记录与重算

coverage.evidence 使用版本 1，保存原计划窗口和每条 valid / missing / rejected / expired 记录。valid 带已接受的任务域曝光时刻，其余记录曝光为 None。正常等待轮询不产生记录。导出的记录逐条复制，修改报告不影响在线累计。

离线复核不实例化 FollowupCoverage，也不读取其累计区间：从相邻记录提取两个有效曝光之间且不超过 250 ms 的边，再排序、求并集并求窗口补集。独立计算覆盖时长、固定分母下的比例、未覆盖长度及最长未覆盖区间，同时核对计数、首末曝光和完成状态。数字比较的绝对容差为 1e-9。

连续缺失记录不能被跨过。若删除中间缺失记录，即使同步修改 missing 计数，重算出来的区间仍会与原摘要不同而被拒绝。无效版本、重复/乱序曝光、非有限指标、窗口错误与摘要矛盾也会被拒绝。

版本和定义可识别的旧 coverage 没有 evidence 时返回 unavailable / legacy_coverage_without_records，不将其当作 verified；空对象或未知版本会被拒绝。unavailable 只说明缺少重算依据，不表示旧摘要已经验证。某个场景错误会记为 rejected，其他场景仍继续核对。

## 工具与结果

```sh
make followup-coverage-audit-test
make followup-coverage-audit
PYTHONPATH=src:. .venv/bin/python scripts/audit_followup_coverage.py \
  docs/evaluations/followup-coverage-evidence-v1.json --output outputs/my-coverage-audit.json
```

CLI 针对后续监测压力报告的 results 逐项检查，并与每项 actual_status 核对窗口完成标志。全部 verified 返回 0；存在 rejected 返回 1；只有 verified/unavailable 且含 unavailable 时返回 2。输入无法读取、JSON 损坏或报告头无效返回 3，不创建输出；输出写入失败返回 4。错误在 stderr 中给出简短诊断。输出路径必须未占用，父目录须存在，已有文件不会被覆盖。

2026-11-03 入口校验扩展：JSON 任意层级的重复字段、NaN/Infinity 和溢出为无穷的指数数字均被拒绝。报告头要求整数版本 1、test_kind 为 deterministic_session_fault_matrix、正整数 cases 与 results 长度一致。逐项状态必须为 completed / cancelled / invalidated；未知状态或未结束场景记为 rejected，其他场景继续复核。首末曝光必须为有限数字，布尔值不能冒充数字；窗口长度溢出也被拒绝。

案例数核对可发现未同步改动 cases 的截断，但不证明矩阵完整：同时删减 results 并修改 cases 仍可能形成一致输入。工具不验证预定参数组合是否全部出现，也不验证报告来源或真实性。

- [带记录的 108 场景报告](evaluations/followup-coverage-evidence-v1.json)：原有状态与告警判定仍为 0 项不一致。
- [独立覆盖复核结果](evaluations/followup-coverage-audit-v1.json)：108 verified。
- 既有不带记录的 108 场景报告在兼容性测试中全部为 unavailable，原文件不改写。

复核仅证明这些记录与覆盖摘要的算术一致性。它不验证 valid 标记本身是否正确、原始图像是否真实，也不能检测一份同时改写了记录和全部摘要的伪造报告。覆盖比例仍不是物理安全指标。

## 验证

新增 8 项测试覆盖 JSON 往返与独立区间并集、完整压力矩阵、摘要篡改、删除缺失记录、记录隔离及顺序、旧报告/提前结束语义、单场景错误隔离和非法版本/窗口/非有限值。

相关回归共 258 项通过（58 核心、31 接触、34 释放、13 线程采集、14 曝光、7 采集进程、8 进程恢复、39 恢复相关、9 独立后续监测、10 进程后续监测、35 后续边界/压力/覆盖与复核），Ruff 与差异检查通过。未连接真实相机，未新增物理抓放或训练。

2026-11-03 新增 6 项输入与 CLI 回归，覆盖报告头/截断、未知状态、缺证据的版本校验、布尔曝光及窗口溢出、损坏 JSON/编码/文件缺失、退出码与已有输出保护。本轮后续监测相关测试共 41 项通过，其中复核测试 14 项；108 场景仍全部 verified，历史报告仍为 108 unavailable。
