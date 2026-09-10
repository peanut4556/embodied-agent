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

## License

Apache License 2.0。参见 [LICENSE](LICENSE)。
