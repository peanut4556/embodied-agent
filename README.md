# Embodied Agent

A safety-oriented starter for building embodied AI agents with **RAI**, **LeRobot**, and
**ROS 2**.

这是一个面向机器人具身智能的 Agent 工程骨架，重点是先建立清晰、可测试的系统边界：

```text
自然语言任务 -> RAI 任务规划 -> 安全校验 -> LeRobot 策略 -> ROS 2 -> 机器人/仿真器
```

当前版本可以在没有机器人、GPU、ROS 2 本机安装或 API Key 的情况下运行完整 mock
闭环，并提供已验证的 ROS 2 Jazzy action/status 消息桥。

> [!IMPORTANT]
> 本项目当前是开发原型，不是经过认证的机器人安全控制器。接入真实机器人前，必须在
> 独立于大模型进程的控制层实现急停、碰撞检测、关节/速度/力矩限制和通信超时保护。

项目全景、两分钟面试讲述与完整实验索引见 [面试讲述材料](docs/interview-guide.md)。

## 当前能力

- 结构化 `Plan`、`PlanStep`、`Observation`、`Action` 数据模型
- 可替换的 Planner、Policy、Robot 接口
- 无外部依赖的规则规划器和 mock 机器人
- RAI planner、LeRobot policy、ROS 2 robot 适配层
- 动作白名单与最大规划步骤限制
- ROS 2 `/embodied_agent/action` 安全校验桥
- ROS 2 `/embodied_agent/status` 执行状态反馈
- Docker 化 ROS 2 Jazzy ARM64 环境
- 本地单元测试、ROS package 测试和 topic loopback 测试

## 系统架构

```text
                     structured plan
User instruction ───────> Planner (RAI)
                               │
                               ▼
ROS observations ───────> Policy (LeRobot)
                               │ normalized action
                               ▼
                         Safety Gate
                               │ validated JSON
                               ▼
                        ROS 2 Bridge ─────> Robot / Simulator
                               │
                               └──────────> status / telemetry
```

详细接口和部署边界参见 [docs/architecture.md](docs/architecture.md)。

## 环境要求

- Python 3.12
- Docker Desktop 或 Docker Engine
- ROS 2 Jazzy（项目默认通过容器提供）
- macOS Apple Silicon 或 Ubuntu 24.04

已经验证的依赖版本：

| 组件 | 版本/环境 |
|---|---|
| Python | 3.12 |
| LeRobot | 0.6.1，独立 `.venv` |
| RAI Core | 2.12.1，独立 `.venv-rai` |
| 本地 LLM | Ollama 0.33.3 + Qwen3.5 4B |
| ROS 2 | Jazzy / Ubuntu 24.04 ARM64 |
| PyTorch | 2.11 |

LeRobot 需要 NumPy 2.x，而 RAI Core 当前需要 NumPy 1.x，因此二者必须保持在独立
Python 环境中，通过 JSON、ROS message 或网络 API 通信。

## 快速开始

```bash
git clone https://github.com/peanut4556/embodied-agent.git
cd embodied-agent

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

embodied-agent "把桌上的红色积木放进盒子"
pytest -q
```

预期动作顺序：

```text
locate -> pick -> place -> verify
```

## 安装 LeRobot 与 RAI

推荐使用 [uv](https://docs.astral.sh/uv/) 重建两个隔离环境：

```bash
# LeRobot / policy 环境
uv sync --extra dev --extra lerobot

# RAI / planner 环境
uv venv --python 3.12 .venv-rai
uv pip install --python .venv-rai/bin/python -r requirements/rai.txt
```

验证安装：

```bash
make deps-check
```

RAI 的模型供应商尚未写死。运行以下命令后，可以选择 OpenAI、Ollama、AWS Bedrock
或其他受支持的模型服务：

```bash
.venv-rai/bin/rai-config-init
```

不要把 API Key 提交到 Git；本项目已经忽略 `.env`。

### 本地免费模型

项目默认的本地 RAI 配置位于 `config/rai.ollama.toml`，使用 Apache 2.0 许可的
`qwen3.5:4b`。模型通过 Ollama 在 `127.0.0.1:11434` 提供服务，不需要 API Key。

安装 Ollama 后，拉取并检查模型：

```bash
make model-pull
make ollama-check
```

用本地模型生成计划并执行 mock 机器人闭环：

```bash
make rai-demo
```

也可以直接运行：

```bash
PYTHONPATH=src .venv/bin/python -m embodied_agent \
  --planner rai "把桌上的红色积木放进盒子"
```

RAI 在 `.venv-rai` 中运行，主程序在 `.venv` 中运行，两边只通过 JSON 交换结构化
任务计划。模型输出仍需经过 `SafetyGate`；本地模型不会获得绕过动作白名单的权限。

## ROS 2

### 可视化机械臂仿真

新增 **MuJoCo 三维物理模式**：`make physics-build` 构建后运行
`make physics-ros`，另一个终端运行 `make sim-web-background`，打开
<http://127.0.0.1:8765>。积木依靠双指接触与摩擦被抓起，释放后受重力落入托盘。
详情与测试命令见 [docs/physics.md](docs/physics.md)。
物理镜像默认用俯视 RGB-D 相机估计红色积木的抓取坐标；定位失败会停止执行。
运行 `make vision-test` 可验证三个不同位置的视觉抓取及异常处理。

可运行 `make record-demo DATASET_ROOT=outputs/datasets/my-first-demo` 采集视觉抓取演示，
生成 LeRobot v3 图像/关节/动作数据集，并自动读回及回放验证。
依赖安装、字段定义和预览说明见 [docs/datasets.md](docs/datasets.md)。

第一版学习基线已支持 `make imitation-train` 和 `make imitation-evaluate`：
用首帧视觉特征学习关节动作序列，并在保留位置进行物理评测。
它是固定场景的开环回归策略，详见 [docs/imitation.md](docs/imitation.md)。

已有模型还可配合视觉与夹持反馈，在目标移动、首次抓取失败后退回观察位重新规划。
运行 `make feedback-evaluate` 可比较开环与反馈执行，流程和限制见
[docs/feedback.md](docs/feedback.md)。

学习反馈执行已接入同一个 ROS 2 控制台。准备好训练权重后运行
`make physics-feedback`，在网页选择“学习模型 · 视觉与接触反馈”。规则或 Qwen 计划
中的抓取、放置会合并为学习技能，界面显示恢复次数、停止原因及原始计划。
启动配置和复验命令见 [docs/feedback.md](docs/feedback.md#ros-2-控制台)。

运输途中也会持续监测夹持：短暂丢失时暂停确认，持续丢失时退回观察位重新抓取，
无法识别落点或超出训练范围时停止。`make slip-evaluate` 使用 MuJoCo 外力拉落积木，
对比开环与反馈执行，详见 [运输滑落验证](docs/feedback.md#运输途中滑落检测与恢复)。

`make recovery-record` 可把正常、滑落恢复、无法恢复及用户停止的全过程保存成
带结果标签的 LeRobot 数据，并自动读回回放验证。成功和失败 episode 分别列出，
不会混进原先的成功示范训练入口。字段、精确回放和使用限制见
[恢复过程数据](docs/recovery-data.md)。

`make temporal-prepare` 将恢复数据整理为连续观测和动作窗口，提供成功动作模仿与
结果标签两个独立入口，检查场景划分和数据完整性。该命令只完成数据准备，原有
恢复批次没有独立测试集。见 [连续反馈训练数据](docs/temporal-data.md)。

首个连续反馈 MLP 基线已完成新示范采集、训练和物理对比，使用
`make reactive-record reactive-prepare reactive-train reactive-evaluate` 复现。
本轮新模型抓放为 0/3，现有反馈控制器为 3/3，因此新模型仅保留作研究基线，
不替换网页执行端。数据、结果与后续改进见 [连续反馈学习基线](docs/reactive.md)。

后续新增带记忆的 GRU 序列模型，并按开发场景实际任务完成数选模。
运行 `make memory-train memory-select memory-evaluate` 可复现；本轮新测试中
记忆模型完成 1/3，现有反馈控制器完成 3/3，仍保留原控制台执行方案。
选模结果、限制和纠正示范方向见 [带记忆模型实验](docs/memory-policy.md)。

已新增 `make correction-record correction-check`，记录学习模型实际执行后的专家纠正
过程，并按稳定夹持、持续提升和主动释放筛选训练目标。首批 7 段中 4 段合格，
其余失败/停止不作为模仿目标。质量复查也确认：上述记忆模型的 1/3 是入盒结果，
合格抓放实际为 0/3；原反馈控制器仍为 3/3。详见 [纠正示范与质量检查](docs/correction-data.md)。

后续已新增 14 条纠正轨迹并完成 600 轮实际微调，三个候选开发合格抓放均为 0/3；
未启用新策略，预留测试未执行。见 [纠正微调实验](docs/correction-finetune.md)。

第二轮已定位到 0.52–0.76 秒的早期持续偏离，新增 14 条提前接管轨迹并再训练；
三个候选合格抓放仍为 0/3，单个候选出现 1/3 稳定抓取但未完成抬升。
见 [早期偏差与第二轮纠正](docs/correction-early.md)。

后续短段对照确认：两个模型在三个开发场景的前两秒都未触发执行器限速，
取消限速的轨迹完全相同。见 [动作链路对照](docs/action-chain.md)。

视觉/关节输入对照已完成：参考关节输入使两轮模型的平均短段状态误差分别下降约 91%/93%，
参考视觉改善不稳定。这是诊断参考输入，不能部署；见 [输入反馈对照](docs/input-feedback.md)。

进一步完成 360 组单帧关节扰动和 18 条闭环短轨迹：腕关节输入最敏感，
第二轮部分位置对 0.01 rad 测量扰动出现明显轨迹变化。见 [关节敏感性实验](docs/joint-sensitivity.md)。

同预算关节扰动增强训练对照已完成：正常输入两组均为 0/3，正向腕关节测量扰动下
增强组为 1/3、对照组为 0/3。未更换策略；见 [增强训练对照](docs/joint-training.md)。

多种子复核新增 101、211 均未复现上述成功，三个种子的正常输入合格抓放均为 0/3；
当前增强配置尚无稳定改善证据。见 [多种子复核](docs/joint-seeds.md)。

同状态下一帧目标验证完成：182 个目标与真实推进一致，局部动作调整使训练/验证
平均状态误差下降约 7.1%/9.0%；该阶段尚未更新策略。见 [状态目标验证](docs/state-targets.md)。

后续局部目标蒸馏已实际完成：101 个训练目标参与同预算对照，验证动作误差降低约 8.8%，
但两组正常输入合格抓放仍为 0/3。见 [局部目标蒸馏](docs/state-distillation.md)。

最新模型的自主执行状态已重新采集：14 条接管轨迹全部由专家恢复，完成 70 个多步关节/
物体目标校验；该阶段目标尚未用于训练。见 [多步物体目标](docs/multistep-targets.md)。

后续完成状态屏蔽与三组训练对照已完成：4 个未来标签被屏蔽，三组验证均为 0/3，
且都未保留父模型的已知训练位置成功。见 [完成状态屏蔽](docs/completion-mask.md)。
后续加入同历史动作冲突审计和成功轨迹回放训练对照，见 [成功轨迹回放](docs/success-replay.md)。
自主轨迹审计定位到初始动作偏离及提前 0.28 秒闭合后的接触中断，见 [首次偏离审计](docs/replay-divergence.md)。
已采集六个接触邻近状态的恢复轨迹，三条合格恢复提供 933 帧专家监督，见 [接触恢复样本](docs/contact-recovery.md)。
连续夹持判定在相同接管点将专家恢复率提高到 4/6，见 [稳定接触对照](docs/contact-stable.md)。
剩余失败的 RGB-D 证据见 [视觉拒绝审计](docs/contact-vision.md)，分别涉及平面旋转尺寸误判和物体倾斜。
可选旋转尺寸判据将同六个接管点专家恢复提高到 5/6，见 [旋转尺寸恢复对照](docs/contact-oriented.md)。
剩余倾斜案例经四秒等待仍不满足视觉条件，见 [受限等待审计](docs/bounded-wait.md)。
新增可选[接触丢失恢复闭环](docs/contact-watchdog.md)：第 79 帧自动接管可恢复原失败策略，正常父策略轨迹不变；原第 84 帧晚接管仍失败。
[位置与视觉泛化评估](docs/policy-generalization.md)完成 48 次配对执行：纯策略 1/24、带恢复闭环 5/24；另发现一次不必要接管，当前泛化仍弱。
进一步支撑接触与静态可达约束见 [支撑与可达审计](docs/support-reach.md)。
RGB-D 倾斜面与可见间隙估计见 [可见几何审计](docs/visible-geometry.md)，可见间隙不代表完整碰撞净空。
已验证带尺寸先验与误差预算的 [保守外形包络](docs/conservative-geometry.md)，尚不构成动作放行依据。
[误差范围审计](docs/geometry-error.md)发现倾斜图像边缘存在约 26 mm 离群点，需先处理点归属与墙面关联。
[过滤与墙面分组验证](docs/rgbd-filter.md)改善点归属，但过滤后的可见间隙仍不可直接用于动作放行。
[组合压力验证](docs/envelope-stress.md)将可靠点接入包络，加入视点、图像遮挡与深度扰动，仍不提供路径安全许可。
[夹爪扫掠检查](docs/gripper-sweep.md)进一步覆盖路点间运动，当前仅用于保守几何筛查。

下面保留无需物理引擎的二维快速演示：

启动 Docker Desktop、Ollama 后，在两个终端分别运行 `make sim-ros` 和 `make sim-web`，
打开 <http://127.0.0.1:8765>。输入默认中文任务，可查看抓取、放置动画、ROS 状态反馈与
积木入盒的位置验收；控制台提供规则/Qwen 规划切换、停止和场景重置。

此版本为二维运动学仿真，尚不模拟接触动力学。启动、通信和限制说明见
[docs/simulation.md](docs/simulation.md)。

构建 ROS 2 Jazzy 镜像并运行 topic 回环：

```bash
make ros-build
make ros-test
make ros-demo
```

回环成功时会输出：

```json
{"accepted": true, "action": "pick", "reason": ""}
```

进入交互式 ROS 2 环境：

```bash
make ros-shell
```

更多说明参见 [docker/README.md](docker/README.md)。

## 完整验收

```bash
make verify
```

该命令依次验证：

1. Python Agent 与适配器单元测试
2. LeRobot 和 RAI 的隔离环境
3. ROS 2 Jazzy、`rclpy`、`ros2cli`、`std_msgs`
4. ROS package 测试
5. action/status topic 回环

## 项目结构

```text
embodied-agent/
├── src/embodied_agent/          # Agent runtime、模型和适配器
├── tests/                       # Python 单元测试
├── ros2_ws/src/                 # ROS 2 workspace 与安全消息桥
├── config/                      # 机器人配置样例
├── docker/                      # ROS 2 Jazzy 镜像
├── docs/                        # 架构与安全边界
├── requirements/rai.txt         # 独立 RAI 环境依赖
├── Makefile                     # 常用开发/验证命令
└── pyproject.toml               # Python 项目与 LeRobot extra
```

## 路线图

- [x] Mock 任务规划、执行与反馈闭环
- [x] LeRobot、RAI、ROS 2 的隔离适配层
- [x] ROS 2 action/status 安全消息桥
- [x] 接入 Ollama + Qwen3.5 本地 RAI LLM planner
- [ ] 接入 LeRobot 预训练策略或 ACT policy
- [x] 恢复过程采集、精确回放与连续观测训练数据入口
- [x] 首版连续反馈模型训练及固定新场景物理对比（未达到可用水平）
- [x] GRU 持续记忆、开发场景物理选模和新测试评测（仍未达到可用水平）
- [x] 模型执行后的专家纠正数据、精确回放和抓放过程质量检查
- [x] 扩充纠正示范并完成带监督屏蔽的 GRU 微调（开发合格抓放仍为 0/3）
- [ ] 定位闭环偏差、扩充视觉条件并通过合格抓放物理评测
- [x] [首次抓持超时恢复](docs/acquisition-recovery.md)：新仿真网格合格抓放 14/16，原恢复器 2/16（专家混合系统，未默认启用）
- [x] [有上限的下降宽限](docs/descent-recovery.md)：成功正对照的不必要接管 5→1，已见网格恢复成绩保持 14/16
- [x] [告警后的有限视觉复查](docs/post-alert-verification.md)：72 个合成场景符合预期，持续稳定新证据才能重新确认（未执行机械恢复）
- [x] [进程采集驱动复查](docs/process-post-alert-verification.md)：已接入主动采集与会话交接，真实子进程返回/卡死/崩溃/停止 4 条路径通过（模拟坐标，未执行机械恢复）
- [x] [按观测失败主动选视角](docs/active-view.md)：12 次物理回放对照中恢复三组被遮挡的视觉确认，完整动作/关节轨迹不变（已知场景，实验入口启用）
- [x] [双视角一致性门槛](docs/cross-view.md)：渲染实验中拦截单侧 20 mm 标定偏差的一次错误确认；共同偏差仍会失效，单侧缺失时拒绝确认
- [x] [固定参考点平移校正](docs/reference-confirmation.md)：仿真中纠正共享 20 mm 偏差并恢复单侧偏差下的正确确认；参考表本身错误仍会失效（未真实标定）
- [x] [独立检查点约束的刚体校正](docs/rigid-reference.md)：渲染对照恢复小角度旋转下的确认，拒绝离面尺度/检查点失配并限制目标外推距离（共同参考错误仍会失效）
- [x] [校正后误差范围确认](docs/uncertainty-confirmation.md)：108 个合成边界对照中，4 mm 预算组错误确认 6→0，代价是少确认 12 个正确案例；预算低估仍会失效（显式启用）
- [x] [按边界余量主动选观测](docs/uncertainty-view.md)：合成正常组正确确认从粗观测 18 增至主动选择 30，切换后必须获取新证据；配置精度虚报仍会误确认（未验证物理视角精度）
- [x] [渲染误差估计与缺失回退](docs/rendered-precision.md)：12 个渲染场景中，粗配置确认 3、仅边界切换 4、加入缺失回退 5，错误确认均为 0；粗配置预算在验证集超界，尚不能作为可靠上界
- [x] [配置证据准入与撤销](docs/profile-admission.md)：排除历史超界配置，新场景中确认 6→2、已排除配置请求 87→0；新证据又暴露高分辨率覆盖不足，合并后无合格配置即不采集（未获可靠性认证）
- [x] [连通区域参考检测](docs/reference-components.md)：修复小亮点使检查点整体被拒的问题；同图像对照中新场景确认 1→8/12，错误确认 0，完整覆盖 4→11/12，在原预算与门槛下恢复准入（仅仿真）
- [x] [颜色身份与深度覆盖分离](docs/reference-depth-coverage.md)：稀疏参考深度断带下恢复 64/64 路校正观测，拒绝无深度第二候选；1 mm 噪声仍有预算超界，未获精度保证
- [x] [带原始证据保护的三帧估计](docs/temporal-reference.md)：新种子渲染对照中，168 路匹配观测超预算 2→0，确认 6/6 保持；代价为延后 0.4 秒，共同偏差仍可误确认
- [x] [长窗口原始范围检查](docs/settled-reference.md)：预设动态渲染中额外拒绝 4 个仍缓慢移动的确认，保留静止和停止后确认；增加 0.4–0.6 秒延迟，不构成零速度或任务安全证明
- [x] [外力动力学与帧间运动审计](docs/force-disturbance.md)：500 Hz 真值揭示低频图像漏检；提供 5/20 Hz 双视角对照，量化采样成本与恢复延迟
- [x] [同预算非等间隔观测](docs/sampling-phase.md)：16 次采样在五相位仿真网格中额外阻止 8 次受扰动确认，保留 4 次静止与恢复确认；仍有 2 次漏检，未替代高频观测
- [x] [随机观测与跨周期验证](docs/random-sampling.md)：三个冻结随机计划保持 16 次预算，两个在本组阻止全部周期扰动确认，一个仍漏检；记录种子差异与固定计划的新反例
- [x] [候选确认后的高频复核](docs/confirmation-challenge.md)：低频候选触发一秒 20 Hz 复核，保留原期限和 31 次上限；本组拒绝两次周期运动确认，实际采集成本与延迟均记录
- [x] [双时钟复核与交付期限](docs/delivered-challenge.md)：126 组故障回放验证延迟、丢帧和元数据拒绝，揭示当前 3 秒协议没有处理延迟余量
- [x] [显式结果交付预算](docs/delivery-budget.md)：采集保持三秒，显式选择 100 ms 结果交付预算；308 组回放检验延迟恢复及超界拒绝，默认仍为零额外预算
- [x] [单请求采集与因果时序](docs/serial-challenge.md)：按请求时刻重新渲染 RGB-D，20/50 ms 虚拟串行服务完成复核，80/100 ms 因曝光间隔拒绝；隔离卡死和迟到回调
- [ ] 支持具体机械臂和相机
- [ ] 增加任务级重规划、超时和失败恢复
- [x] 增加 MuJoCo 仿真环境与 Qwen → 学习反馈执行的端到端评测

## 安全边界

当前 ROS 2 bridge 只验证语义动作格式与白名单。真机部署还必须实现：

- 硬件急停和 watchdog
- 关节、速度、加速度、力矩与工作空间限制
- 自碰撞和环境碰撞检测
- 观测时间戳、动作序列号和过期动作拒绝
- 通信断开后的安全停止
- 每一步的成功/失败检测与审计日志

最新实验：[有界流水线曝光调度](docs/pipelined-challenge.md)，验证固定处理延迟下的容量约束、按序回调和取消边界（合成时序实验）。

最新扩展：[流水线抖动验证](docs/pipeline-jitter.md)，用 192 个合成案例分离处理延迟波动与曝光抖动，并定位原复核协议的时间余量限制。

最新实验：[显式采集余量协议](docs/spaced-challenge.md)，通过独立的 3.16 秒曝光期限容纳 0–5 ms 迟到，保留原运动判据；128 例合成对照已完成。

最新审计：[采样周期与相位盲区](docs/sampling-aliases.md)，390 个合成案例保存固定采样节奏的运动漏检反例；采集余量不等于消除混叠。

最新实验：[有界非等间隔曝光](docs/unequal-sampling.md)，4,428 例配对扫描中减少了部分混叠误确认；代价是仅容忍 2 ms 迟到，残余反例仍保留。

最新实验：[累积相位曝光计划](docs/phase-sweep.md)，同预算下拒绝上一轮 12 个残余混叠反例，并完成独立周期/相位网格与原网格回归；仍限于合成验证。

最新审计：[运动波形与返回前观测空窗](docs/motion-shapes.md)，1,158 个合成案例明确窄脉冲、晚启动运动和曝光后新运动的限制。

最新扩展：[两阶段新鲜证据门槛](docs/fresh-witness.md)，候选后补充五对双视角观测并显式报告证据时效；低延迟能力仍是合成实验假设，窄脉冲盲区仍保留。

最新扩展：[两阶段统一因果采集接口](docs/two-stage-capture.md)，把候选、请求、最终确认与取消纳入同一个可审计状态机；56 条虚拟服务运行已验证，真实相机适配仍待完成。

最新扩展：[两阶段真实线程运行适配器](docs/two-stage-runtime.md)，有界采集、取消及线程回收状态已接通；真实时钟与合成读取完成一轮完整流程，真实相机尚未接入。

最新扩展：[可终止双视角采集进程后端](docs/process-pair-capture.md)，预热隔离进程接入两阶段运行器，验证阻塞读取消升级与退出回收；真实设备清理仍需单独确认。

最新扩展：[显式采集恢复与会话隔离](docs/capture-recovery.md)，核实旧进程退出后才启动新会话；保留恢复后完整通过及真实调度超时拒绝的两轮记录。

最新扩展：[跨进程采集时间链诊断](docs/capture-timing.md)，通过不同位置的延迟注入区分“曝光不及时”和“曝光后返回较慢”；提供分段实测证据，未改变确认门槛或宣称实时保证。

最新扩展：[提前预约曝光原型](docs/scheduled-capture.md)，在合成读取器的 5 ms 派发延迟条件下，满足 2 ms 曝光窗口的数量由到点派发的 0/36 提升为提前预约的 36/36；正常条件仍有漏窗，尚未接入任务确认流程。

最新扩展：[预约式两阶段完整确认](docs/reserved-two-stage.md)，提前预约已接入 31 次候选采集与 5 次新鲜见证；5 ms 派发延迟下两次真实进程合成运行一次完整通过、一次按时序拒绝，仍无实时稳定性保证。

最新扩展：[预约恢复与一次性近期证据交付](docs/recovery-handoff.md)，故障回收后重建预约会话，领取时按实际曝光时间复核时效；真实进程实验验证立即领取、重复拒绝与延迟过期，交付不等于机械动作授权。

最新扩展：[证据队列消费边界](docs/evidence-inbox.md)，在下游回调入口复核排队后的时效与会话撤销，回调异常标记结果未知并禁止自动重放；8 次真实进程合成联调覆盖立即交付、过期、会话切换及回调异常。

最新扩展：[崩溃后的证据账本对账](docs/evidence-ledger.md)，持久化唯一证据编号，真实进程故障注入区分提交前退出与提交后回执丢失；保留结果未知的原始审计，不自动重放消费。

最新扩展：[调用方崩溃后的证据发现](docs/indexed-evidence-ledger.md)，先持久化提交索引，重启后无需原进程保留编号即可分页对账；10 次进程退出实验覆盖索引与证据事务边界，恢复只读且不自动重放。

最新扩展：[数据库身份与历史检查点](docs/evidence-checkpoint.md)，恢复前核对独立保存的历史基线，拒绝错误数据库和已知历史丢失；18 个恢复案例包含两个明确验证的未覆盖尾部丢失限制，检查点不构成动作授权。

最新扩展：[可事务更新的独立检查点仓库](docs/checkpoint-store.md)，以版本比较更新和 SQLite 事务保存恢复基线，拒绝过期更新覆盖新版；6 次真实进程退出实验验证提交前后恢复，仍需独立保管仓库。

最新验证：[多进程检查点竞争与锁持有者退出](docs/checkpoint-contention.md)，6 组真实进程实验验证同一版本只有一个发布者成功，以及锁持有者退出后的显式恢复；18 个工作进程全部回收。

最新扩展：[只读恢复诊断命令](docs/evidence-audit.md)，将历史核验与索引对账整合为 JSON 报告和分级退出码，明确区分核验通过、未落账、未扫描完和故障阻断。

最新修复：[双向证据一致性审计](docs/bidirectional-audit.md)，新增反向检查，拒绝旧单向扫描会漏掉的无索引证据；反向未扫描完也不能返回通过。

最新修复：[证据记录合同校验](docs/evidence-semantics.md)，在摘要与双向索引一致之外复核正文身份、状态和原始接收期限；三类摘要一致的异常均被诊断阻断。

最新扩展：[持久化检查点覆盖诊断](docs/checkpoint-coverage.md)，有效但未覆盖的数据返回独立退出码 6；完整验证发现覆盖缺口、显式发布和旧副本回退阻断流程。

最新扩展：[严格检查点发布准入](docs/verified-publication.md)，同一源库快照内先核对完整证据再生成检查点，异常或超过检查上限时保留旧基线；新增显式严格发布接口。

最新扩展：[绑定预览内容的检查点发布](docs/reviewed-publication.md)，审核后即使只是合法追加也会使旧提案失效；只有匹配原提案内容与仓库版本才能发布。

最新扩展：[跨进程检查点提案命令](docs/checkpoint-cli.md)，预览输出可保存的 JSON，发布显式读取原提案；不同退出码区分成功、过期提案、请求失败与发布结果未确认。

最新扩展：[提案发布事务对账](docs/publication-receipt.md)，发布记录与检查点同事务提交，成功回执丢失后可按原提案只读查询历史结果，无需重放。

最新联调：[真实进程采集到检查点发布](docs/capture-publication.md)，两次立即路径完成采集、落账、跨进程发布与对账，两次延迟路径按过期拒绝；读取器仍为合成数据。

最新联调：[消费者结果未知的完整链路对账](docs/capture-consumer-faults.md)，落账后回调报错与消费前过期各两次均按持久化结果分流且不重放；整轮 7/8，保留一次采集时序拒绝。

最新联调：[采集调用方退出后的无编号恢复](docs/capture-owner-exit.md)，真实采集确认后强制退出调用方，恢复程序仅凭路径发现索引；提交前后各两次验证未落账拒绝和已落账提案生成。

最新扩展：[调用方死亡监测](docs/owner-watchdog.md)，可选独立存活连接让阻塞采集工作进程在调用方突然退出后自行终止；4 次真实进程实验通过，仍不保证原生 GIL 死锁或硬件清理。

最新原型：[独立采集作业监督进程](docs/capture-supervisor.md)，工作进程被 SIGSTOP 暂停且调用方突然退出时仍可终止回收；两次真实实验通过，尚未接入完整采集后端。

最新扩展：[独立监督采集后端](docs/supervised-capture.md)，独立监督已接入实际请求/帧传输与预约确认；两次完整运行均完成 36 请求，双层进程退出可核验，默认模式未变。

最新扩展：[监督进程失联保护](docs/supervisor-loss.md)，监督进程被 SIGKILL 后阻塞读取进程可自行退出；四次真实实验通过，内部仍保留清理未确认，避免误启动新会话。

最新扩展：[独立采集运行时限](docs/capture-runtime-limit.md)，调用方仍活着但被暂停时，监督进程也能按显式时限终止作业；两次实际暂停实验通过，默认不启用。

最新验证：[独立时限后的会话重建](docs/supervised-recovery.md)，两次均完成旧进程回收及新会话隔离，但恢复后的完整确认 0/2，分别在第 5/15 请求错过曝光窗口；失败记录完整保留。

最新实验：[恢复后曝光末段自旋](docs/recovery-spin.md)，新增可选等待策略并做 A/B/B/A 对照；默认休眠与 1.5 ms 自旋各 2/2 通过，尚未证明改善，默认关闭。

最新测量：[预约等待 CPU 成本与迟到](docs/wait-cost.md)，新增成功/失败路径逐请求遥测；本轮两种策略均 0/2 完整通过，自旋观测请求的平均等待 CPU 约为休眠 6.7 倍，未证明收益。

最新测量：[固定分母曝光窗口对照](docs/fixed-wait-probe.md)，两种策略各测满 72 个独立窗口均通过；自旋最大目标迟到较小，但平均等待线程 CPU 约 9.8 倍，默认继续关闭。

最新测量：[后台 CPU 负载下的固定窗口对照](docs/wait-under-load.md)，288 个窗口完整记录，基线休眠 69/72、自旋 72/72；两个后台负载进程下两者均 72/72，自旋等待 CPU 成本仍明显较高，尚不支持默认启用。

最新测量：[平衡顺序的等待策略重复实验](docs/balanced-wait-campaign.md)，预先固定 4 个区组、576 个窗口，平衡条件位置与区组内前后顺序；两种负载条件下休眠均 143/144、自旋均 144/144，成本增加且收益证据仍有限，不自动启用。

最新扩展：[等待 CPU 预算自动回退](docs/wait-cpu-budget.md)，读取器累计等待 CPU 达到显式预算后，后续请求停用自旋；216 个窗口全部交付，预算策略等待 CPU 比持续自旋少约 76%，仍高于纯休眠，默认不启用。

最新联调：[跨恢复代的等待预算分配](docs/recovery-wait-budget.md)，总回退阈值预先均分、失败不退还，避免每次恢复重新领取完整额度；两次初始化阻塞后的新会话均完成 36 请求并从第 5 请求回退休眠，仍为合成读取器验证。

最新扩展：[持久化等待预算池](docs/persistent-wait-budget.md)，启动前事务分配，管理器重建不重置同一池额度；4 进程争用 2 份额度恰好 2 次获配，提交后调用方退出也不退款。预算池不代表设备所有权或硬 CPU 限额。

最新联调：[跨应用进程的持久化预算恢复](docs/persistent-recovery.md)，两轮均在初始化失败、确认清理、应用退出后，由新应用领取剩余份额并完成 36 请求及一次性证据消费；第三个应用在采集启动前因耗尽拒绝。

最新扩展：[预算池身份绑定](docs/wait-budget-identity.md)，重启时可按独立保存的预期池编号打开数据库，拒绝同路径替换的新池；已有异常分配记录也在打开和分配时拒绝。

最新扩展：[预算历史检查点](docs/wait-budget-checkpoint.md)，独立保存历史摘要，重开及分配事务核对已有前缀；正常追加可继续，同池旧副本和已有历史改写被拒绝。

最新扩展：[绑定已查看状态的预算分配](docs/reviewed-wait-budget.md)，在同一写事务中核对完整检查点再分配；两个实际进程争用同一快照时仅一个获配，另一个因状态过期拒绝。

最新扩展：[按已查看预算启动恢复任务](docs/reviewed-recovery.md)，严格模式要求每代显式检查点；遗漏不消耗次数，过期在采集后端创建前拒绝，管理器不自动刷新快照。

最新扩展：[预算分配结果只读对账](docs/wait-budget-lookup.md)，回执丢失后按原请求编号查询历史分配，命令退出码区分已分配、未找到及无法确认，不重放采集。

最新扩展：[启动意图持久化与无编号对账](docs/budget-start-intent.md)，申请额度前保存会话编号和池身份；应用退出后仅凭意图文件查询原分配，不自动重放采集。

最新扩展：[意图文件驱动的恢复对账命令](docs/intent-lookup-cli.md)，仅传文件路径即可对账，可附独立历史检查点；覆盖范围内的数据库回滚返回无法确认，避免被误报为未分配。

最新扩展：[批量恢复审计](docs/budget-recovery-audit.md)，显式清单最多 64 个意图，逐条保留已分配、未找到、无法确认；单条损坏不丢失其他结果，检查点缺失不降级查询。

最新扩展：[同池一致快照对账](docs/budget-snapshot-lookup.md)，一组请求和报告检查点在同一读事务中生成，读取期间发生的新分配不会混入当前批次。

最新扩展：[意图清单自动分组快照](docs/grouped-budget-audit.md)，同数据库、池身份和基线内容的条目合并为一次快照查询，逐文件保留结果并显式报告组边界。

最新加固：[恢复文件无歧义解析](docs/recovery-json.md)，统一拒绝重复字段、非标准数值和过深嵌套，单条对账检查点也执行大小限制。

最新主链路修复：[逐步观测确认](docs/runtime-confirmation.md)，基础 AgentRuntime 不再把无异常调用等同成功；抓放后核对符号观测，失败停止后续步骤，策略不得替换已校验的动作目标。

最新主链路扩展：[有时限的动作确认等待](docs/runtime-wait.md)，异步观测可在期限内确认，等待只轮询不重执行动作；支持取消和读取次数上限，默认立即核验。

最新主链路扩展：[观测新鲜度合同](docs/runtime-freshness.md)，可选要求确认观测采于动作返回之后且未过龄，旧成功状态重放不能直接确认；提供带时间戳 mock 模式。

最新主链路扩展：[连续新鲜样本确认](docs/runtime-multisample.md)，重复帧不累计成功，反例重置、时间戳倒退拒绝；同时修复新鲜度 CLI 参数未传入运行时的遗漏。

最新主链路扩展：[带时间跨度的动作确认](docs/runtime-span.md)，同时约束样本数量、取样跨度和最大间隔，密集短突发或长空档不能直接满足确认。

最新主链路扩展：[结构化确认诊断](docs/runtime-reports.md)，TaskResult 返回每步观测与重置计数、耗时和结束原因，CLI --json 可直接导出供分析。

最新主链路验证：[确认策略故障矩阵](docs/confirmation-faults.md)，三种配置对照六个固定传感器反例；误接受数分别为 4、2、0，全部单次执行。仅为确定性合成合同验证。

最新压力测试：[确认策略的随机化合成故障](docs/confirmation-stress.md)，480 个配对运行揭示时间确认仍接受较长突发与持续假观测，并漏掉部分延迟真结果；保留全部失败，不调整默认参数。

最新主链路扩展：[第二路见证确认](docs/runtime-witness.md)，可选见证回调与主观测同时核验结果、新鲜度和时间偏差；持续主源误报可被冲突见证阻止，尚未接真实第二传感器。

最新双源评测：[见证质量与共同错误对照](docs/witness-stress.md)，640 个配对合成运行展示理想参考、随机错误参考及复制主源的不同结果；共同错误无改善，随机错误增加有效场景漏检。

最新主链路扩展：[见证回调进程隔离](docs/process-witness.md)，阻塞见证可超时退出并回收工作进程，运行时明确报告 witness_timeout，不降级接受单源成功；53 项相关测试通过。

最新故障闭环：[见证进程故障诊断](docs/process-witness.md)，进程退出、读取异常和非法返回转为结构化任务失败；整任务测试验证不执行后续动作，54 项相关测试通过。

最新控制能力：[见证等待可取消](docs/process-witness.md)，进程响应等待分段检查停止请求，取消时回收见证并阻止后续动作；56 项相关测试通过。

最新期限控制：[确认剩余预算传递](docs/process-witness.md)，进程见证按自身超时与任务剩余期限的较小值等待，区分任务超时与见证超时；59 项相关测试通过。

最新资源诊断：[见证进程回收报告](docs/process-witness.md)，逐步报告增加进程生命周期快照，区分接口关闭与回收确认；61 项相关测试通过。

最新任务控制：[跨步骤任务时间预算](docs/runtime-task-budget.md)，规划、动作与确认共享任务期限，耗尽后停止派发新动作；66 项相关测试通过。

最新自动化接入：[任务预算命令行与退出码](docs/runtime-task-budget.md)，--task-timeout 可直接约束任务预算，成功/任务失败/参数错误分别退出 0/1/2；69 项相关测试通过。

最新端到端验证：[跨阶段预算压力实验](docs/task-budget-experiment.md)，240 次配对合成运行中参考期限后派发从 235 次降为 0，但同步调用仍造成最长约 144 ms 返回超期；71 项相关测试通过。

最新动作准入：[最低剩余步骤预算](docs/runtime-task-budget.md)，任务尚未超期但不足调用方指定预留时间时，拒绝派发下一动作并报告原因；74 项相关测试通过。

最新准入评测：[预留时间的收益与误拒绝](docs/admission-budget-experiment.md)，400 次配对合成运行揭示 50 ms 预留消除未确认派发的同时误拒绝 11/40 可行任务，保留取舍且不改默认值；77 项相关测试通过。

最新预算优化：[按动作独立校准](docs/action-budget.md)，合成验证误拒绝由 50/95 降到 21/95，已派发未确认仍为零；提供样本不足和耗时异常回退，81 项相关测试通过。

最新漂移验证：[顺序耗时漂移与回退边界](docs/budget-drift.md)，1080 次配对运行显示保留状态减少未确认动作但增加误拒绝，退化后仍有 30 次未确认；84 项相关测试通过。

最新可选策略：[耗时异常后暂停同类动作](docs/action-budget.md)，顺序评测中退化后未确认从 30 降至 0，但误拒绝显著增加，因此默认不启用；87 项相关测试通过。

最新恢复闭环：[单动作重新校准](docs/action-budget.md)，校验新样本后显式生成预算新版本，仅恢复目标动作，保留其他隔离状态及校准记录；90 项相关测试通过。

最新恢复验证：[独立样本门槛与顺序恢复实验](docs/validated-budget-recovery.md)，360 次合成运行中保留 45 次恢复成功并阻止一次失配恢复；完善批次复用、失败原子性及测量校验，95 项相关测试通过。

最新状态恢复：[预算检查点与只读检查](docs/budget-checkpoint.md)，跨进程保留动作隔离和批次历史，严格校验损坏输入并原子替换文件；103 项相关测试通过。

## License

Apache License 2.0。参见 [LICENSE](LICENSE)。
