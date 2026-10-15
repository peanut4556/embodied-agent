# 外部观测数据契约与离线导入

本轮增加版本 1 的严格 JSON 数据契约和离线导入命令，不修改现有求解器或外部精度门槛。
导入器只校验并规范化数据，不执行拟合、可靠性放行、训练或校正。

**`validated` 仅表示数据结构、声明的一致性和证据文件完整性通过检查。**
导入器不验证文件内容是否真实，不认证传感器性能或物理独立性；报告始终输出
`evidence_contents_authenticated=false`、`calibration_certified=false`、`correction_applied=false`。
策略文件必须由调用者独立选择，不能把观测生产者自行提供的任意策略当成可信授权。

## 文件组成

完整可运行的示例在 [examples/external-observations](../examples/external-observations/)。
示例明确标记为 `synthetic`；证据文件也是测试夹具，不是真实测量证书。

- `observations.json`：`schema_version=1` 和非空 `observations` 数组，最多 10000 条。
- `policy.json`：身份、坐标约定、有效期、误差预算及六类证据引用，所有字段必填。
- `evidence/`：显式指定的本地证据根目录；不访问网络或自动寻找其他凭据。

JSON 重复键、NaN/Infinity、未知字段、缺失字段、非整数版本号、布尔值冒充数值等均拒绝。
命令限制输入/策略各不超过 10 MB。不会用 0.5 mm 或零系统误差补齐缺失预算。

## 坐标与身份

每条观测必须与策略精确匹配：

| 字段 | 约定 |
| --- | --- |
| `data_kind` | `synthetic` 或 `measured`，不能混用；标签本身不证明来源 |
| `sensor_id`, `camera_id` | 指定的外部传感器与相机 |
| `frame_id`, `clock_id` | 已完成上游坐标转换及 UTC 时钟映射后的标识 |
| `mount_id` | 安装配置版本，变化时必须更换策略和证据 |
| `measurement_point` | 首版只接受 `camera_origin` |
| `position_semantics` | 只接受 `translation_from_nominal_camera_origin` |
| `reference_origin_id` | 名义相机原点的版本标识 |

策略另含 `nominal_camera_origin_m`，为指定坐标系中名义原点的三个有限坐标。
输入 `position` 是相机原点相对此名义原点的三维平移量，**不是世界绝对位置或标记点位置**。
示例名义原点为 `[0.4, 0, 1.2] m`；输入 `[1, -2, 3] mm` 规范化为
`translation_m=[0.001, -0.002, 0.003]`，不是把世界位置直接传入既有平移门槛。

每条记录另含唯一的 `observation_id`、`measurement_time`、`camera_time`、`units` 和 `position`。
只支持显式 `m`/`mm`，必须完整给出 xyz。首版不自动进行旋转、安装外参转换或单位猜测；
这些上游转换的剩余误差必须由策略预算及证据覆盖。

## 时间与预算

时间必须是带时区的 ISO 时间，统一到 UTC。调用者必须显式提供 `--as-of`，作为本次离线
检查时刻；历史回放可以指定历史时刻，但这样的报告不是当前可用性证明。

策略必填 `valid_from`、`valid_until`、`reference_time`；`reference_time` 是漂移预算的起点。
`--as-of` 必须位于策略有效期内且不早于漂移起点。观测和相机时间必须处于有效期、
不早于漂移起点、不晚于检查时刻；加入时钟误差后也不能跨越这些边界。

`clock_error_bound_s` 必须同时覆盖时间戳相对 UTC 的残余误差和两观测时间差的残余误差，
具体映射和联合界由同步证据支持。若已有设备只给出各时钟单独误差，调用者必须先形成
满足该约定的保守联合界，不能直接误用其中一个时钟的误差。

计算规则：

- 有效时间差 `skew = abs(measurement_time − camera_time) + clock_error_bound_s`，必须 ≤`max_skew_s`。
- 最坏数据年龄 `age = as_of − min(measurement_time, camera_time) + clock_error_bound_s`，必须 ≤`max_age_s`。
- 漂移时间 `drift_age = max(measurement_time, camera_time) − reference_time + clock_error_bound_s`。
- 系统误差界 = `static_error_bound_m + mount_error_bound_m + frame_error_bound_m + speed_bound_m_s × skew + drift_rate_bound_m_s × drift_age`。

噪声模型首版只接受声明的 `isotropic_gaussian`，`sigma_m` 与上述预算均须为有限非负数。
噪声标准差和系统误差界分开输出，不把漂移/安装误差转成独立高斯噪声。
示例有效时间差为 0.025 s，系统误差界为 **0.000710025 m**；该值只演示计算，不是实测预算。
超大但结构有效的有限预算不会被当作“精度合格”；后续离线门槛仍必须独立检查精度。

## 证据与拒绝行为

策略 `evidence` 必须提供 `noise`、`geometry`、`synchronization`、`motion`、`drift`、`independence`
六类引用，每类包含相对 `path` 和精确 `sha256`。有效期统一由策略声明；证据文件的内容和
有效期限必须由策略制定者审查，导入器不自动解读证书文本或验证签名。示例六类引用同一
份明确标注 synthetic 的文件，只测试完整性流程，不宣称具备六份独立证据。

不存在、哈希不符、绝对路径、`..` 路径以及解析后逃出证据根目录的符号链接都会拒绝。
哈希一致只能说明文件没有偏离策略引用版本，不能证明它支持相应数值或独立性声明。

批次采用全部通过才输出的规则：任一记录无效或 ID 重复，整个批次 `status=rejected`，
`normalized=[]`，同时记录错误索引和原因。不会静默跳过坏记录后输出部分可用数据。
成功时输出 `eligible_for_offline_check=true`；该标志仅供后续离线评估使用，不是运行期校正许可。
命令退出码成功为 0，拒绝为 2。报告包含输入、策略和实现文件哈希；拒绝也留下可审计结果。
报告路径不能覆盖输入/策略路径或落在证据目录内。

## 使用与验证

从仓库根目录运行：

```sh
make external-observation-contract-demo
```

固定历史示例时刻是 `2026-10-14T10:00:01+08:00`。输出位于 `outputs/external-observation-import.json`，
已保存一份 [示例导入报告](evaluations/external-observation-import-example-v1.json)。

真实离线输入需显式指定全部参数：

```sh
PYTHONPATH=src:. .venv/bin/python scripts/import_external_observations.py \
  --input /path/to/observations.json \
  --policy /path/to/reviewed-policy.json \
  --evidence-root /path/to/evidence \
  --as-of 2026-10-14T10:00:01+08:00 \
  --report outputs/import-report.json
```

此处路径和时间仅为语法示例，必须换成待评估批次与审查时刻；仓库没有真实设备证据包。

`make external-observation-contract-test external-reference-test external-reference-stress-test`
共 **26 项测试通过**，包括 13 项新导入测试：单位及预算计算、身份和原点语义、有效期/未来/陈旧数据、
时区、时钟联合界、非法数值、JSON 重复键、证据缺失/篡改/路径逃逸、重复记录、批次原子性、
CLI 成功/拒绝退出码及输出路径保护。Ruff、格式与 Git 差异检查通过。

## 后续边界

下一步可把已校验的观测与对应视觉拟合做严格配对的离线联合报告，验证同一相机、名义原点、
坐标系和采样时刻，保留“导入通过”与“精度检查通过”的区别。在拿到真实观测和独立审查的
预算证据前，只能继续使用明确标注的合成样例，不能宣称真实设备验证完成。

本轮未训练、未运行预留策略测试集、未执行新恢复动作、未应用校正；原恢复成绩仍为 5/6。

后续已完成 [外部观测与视觉拟合的离线联合报告](external-joint-report.md)：重校验证据、严格按 ID/时间/坐标配对，并从原始深度重新计算三个检查，保留导入通过与精度通过的区别。
