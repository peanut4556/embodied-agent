# 外部观测与视觉拟合的离线联合报告

本轮把原始外部观测导入、视觉数据配对和精度检查接成离线流程。
不信任外部输入中的 `validated`、`passes` 或已有拟合参数：重新导入观测并校验证据哈希，
再从前三个参考面的深度数据重新拟合，第四面用于诊断。

整个过程不训练、不写入相机校正，也不认证物理独立性或证据内容。
所有报告始终保留 `calibration_certified=false`、`correction_applied=false` 和
`evidence_contents_authenticated=false`。

## 输入与严格配对

外部输入沿用 [数据契约](external-observation-contract.md)：原始观测、独立选择的策略、
本地证据目录和显式 `--as-of`。不接受旧导入报告来绕过本次有效期或证据检查。

新增视觉 JSON，`schema_version=1`，所有字段必填，拒绝未知字段：

| 字段 | 内容与校验 |
| --- | --- |
| `metadata` | `data_kind`, `camera_id`, `frame_id`, `clock_id`, `reference_origin_id`, `position_semantics`, `nominal_camera_origin_m` |
| `rays` | 7–1000 个有限三维射线，采用原模型的 z=−1 归一化约定 |
| `planes` | 四个名义参考面，逐个包含单位 `normal` 和米制 `offset_m` |
| `calibration` | `start_time`, `end_time`, `depths_m`；32–1024 次四面重复曝光组 |
| `fits` | 逐项 `fit_id`, `observation_id`, `camera_time`, `depths_m` |

标识和坐标语义必须与外部策略一致，名义原点数值也必须一致。现有求解器硬编码的名义
相机原点是 `[0.4, 0, 1.2] m`；首版明确拒绝其他原点，即使输入两侧自称相互一致，
也不会悄悄按另一个原点计算。通用坐标转换不在本轮范围。

每个 `depths_m` 验证组必须为 `4 × 射线数`，校准数组为 `重复数 ×4 × 射线数`，
均使用正的有限米制深度，拒绝字符串和布尔值冒充数值。

外部 `observation_id` 和视觉 `fit_id` 必须分别唯一，且每个外部观测恰好对应一个视觉拟合。
缺项、多项、未知 ID 或重复引用都会整批拒绝，不按数组顺序配对、不挑最近时间戳，
也不静默丢弃无法配对的数据。两侧 `camera_time` 必须代表精确相同的时刻；等价时区表示允许。

校准时间必须有序、处于外部策略声明的有效时间窗口内，并严格早于每个视觉采样时刻。
这只是元数据及时间范围检查，不能证明重复曝光真正独立、噪声长期平稳或时间戳可信；
这些仍是数据来源与评估模型的假设。

任一结构、身份或配对错误：`status=rejected`、`results=[]`。不会在部分记录存在错配时
继续对其余记录给出精度通过结果。

## 重新计算的检查

完成全部配对后，共用原重复曝光估计器和方差上界，从校准数组估计视觉噪声。
每条视觉数据用冻结的候选求解器重新拟合前三面，并执行以下三个独立检查：

1. 第四面残差检查：原插件噪声估计下的名义双侧 1% 门槛。
2. 视觉平移精度检查：校准失败预算 0.005、参数失败预算 0.045，保守平移半径 ≤5 mm。
3. 外部位置检查：使用本次导入计算的噪声标准差与完整系统误差界，外部失败预算 0.01，误差上界 ≤5 mm。

只有数值可用且三个检查都通过，才记录 `passes_offline_checks=true`。这仍只是离线检查结果，
不是校正许可；门槛原有的模型、预算和真实性限制全部保留。

求解不可辨识、停滞或数值计算不可用时，逐例为 `unknown`，不会把默认参数当成通过结果。
输入结构错误与数值未知分别保留；校准噪声计算出现非有限值时整批拒绝。

## 报告状态与退出码

| 层级 | 状态 |
| --- | --- |
| `import.status` | `validated` 或 `rejected`，保留规范化观测与证据完整性检查结果 |
| `pairing_status` | `not_evaluated`, `rejected`, `matched` |
| 顶层 `status` | `rejected` 表示流程输入不可用；`evaluated` 表示完成了离线计算 |
| `results[].status` | `passed`, `rejected`, `unknown` |

CLI 退出码 **0 表示计算完成，不表示所有案例通过**；输入/配对拒绝返回 2。
调用方必须读取 `summary` 和逐例状态，不能用进程退出码批准校正。
报告记录原始输入、策略、视觉文件及相关实现 SHA-256，保留求解结果、噪声上界和三个检查明细。
文件完整性不等于来源认证。

## 合成示例与复现

[示例目录](../examples/external-joint/) 包含两个原始视觉案例及配对外部观测。
它复用上一轮明确标记为 synthetic 的策略和证据。生成器
`scripts/generate_external_joint_example.py` 使用种子 240000、49 条射线和 128 次校准重复，
视觉像素/曝光噪声均设为 0.05 mm；外部示例坐标直接使用合成相机平移，作为流程夹具。
它不是新传感器精度测量或统计性能实验。

```sh
make external-joint-demo
```

示例采用历史回放时刻 `2026-10-14T10:00:01+08:00`，输出
`outputs/external-joint-report.json`；保存的 [示例报告](evaluations/external-joint-example-v1.json)
显示两例均导入通过、配对成功，但最终结果不同：

| 案例 | 第四面残差检查 | 视觉精度半径 | 外部误差上界 | 联合结果 |
| --- | --- | ---: | ---: | --- |
| 正常 | 通过 | 0.393 mm | 3.379 mm | passed |
| 四个参考面共同向 x 偏移 8 mm | 通过 | 0.393 mm | 11.447 mm | rejected |

这明确区分了“导入通过”和“精度检查通过”。门槛函数不读取生成器真值，只有原始观测与声明预算
进入计算。原恢复成绩不因这两个流程样例而增加。

通用命令（路径和时间需换成实际审查批次）：

```sh
PYTHONPATH=src:. .venv/bin/python scripts/evaluate_external_joint.py \
  --observations /path/to/observations.json \
  --policy /path/to/reviewed-policy.json \
  --evidence-root /path/to/evidence \
  --visual /path/to/visual.json \
  --as-of 2026-10-14T10:00:01+08:00 \
  --report outputs/joint-report.json
```

输入文件各限 20 MB；输出不得覆盖源路径或位于证据目录内。整个命令只读输入并写离线报告。

## 验证与下一步

`make external-joint-test external-observation-contract-test external-reference-test fit-stop-test`
共 **37 项测试通过**。新增 11 项测试覆盖正常与共同偏移对照、ID 重排与错配、元数据/原点
不一致、精确时间匹配、校准时间窗口、非法数组、伪造通过标志、从第四面原始数据重算、
不可辨识结果保留未知、重新核对证据和 CLI 状态/来源哈希。

下一步是在获得真实采集批次与独立审查的预算证据后做只读联合回放。当前没有真实设备证据，
本轮不宣称现场验证完成，也不应把这些合成预算替换为默认生产规格。
未训练、未运行预留策略测试集、未执行新恢复动作、未应用校正；原恢复成绩仍为 5/6。
