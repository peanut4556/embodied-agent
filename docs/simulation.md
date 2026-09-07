# 可视化桌面仿真

这是运行在 Mac + Docker 上的二维两连杆运动学演示。输入中文任务后，本机 RAI/Qwen
生成结构化计划；ROS 2 仿真节点按预设轨迹执行，浏览器展示观测 Topic 中的实际状态。

## 启动

先启动 Docker Desktop 和 Ollama（已安装 `qwen3.5:4b`）。在项目根目录的两个终端中分别运行：

```bash
make sim-ros
```

```bash
make sim-web
```

也可用 `make sim-web-background` 独立后台启动控制台，日志和 PID 保存在
忽略提交的 `outputs/sim-web.log` 和 `outputs/sim-web.pid`。关闭它时先核对 PID
对应 `embodied_agent.sim_app` 进程，再使用 `kill <PID>`。

打开 <http://127.0.0.1:8765>，保留默认任务“把桌上的红色积木放进盒子”，点击执行。
“规则计划”可在无需模型的情况下测试执行链路。Qwen 规划可能需要 20–180 秒，
场景运动约 5 秒。任务失败后查看执行记录，点击重置恢复初始场景。

如提示容器名称已占用，说明已有演示容器；无需重复启动。只需启动 `sim-web` 并打开网页。

## 通信与完成条件

```text
浏览器 :8765 → 本机 RAI 子进程 → SafetyGate
          → HTTP 网关 :8766 → /embodied_agent/sim/action
          → ROS 仿真节点 → /embodied_agent/sim/status（指令编号、成功/失败）
                       → /embodied_agent/sim/observation（20+ Hz 场景状态）
          ← 网关状态订阅 ← 浏览器画面与任务验收
```

每条运动指令带 UUID 和 20 秒有效期，状态等待有超时限制。成功反馈在轨迹执行完毕后
发出。模型输出的 `stop` 终止后续计划；用户停止还会取消同一任务尚未执行的指令。
任务最终仅在红色积木位置落入盒子区域且夹爪松开时显示成功。
这比原 mock runtime 的“发完全部动作就成功”多了一层独立位置验收。

只映射本机端口；无需云端模型、外部前端资源或额外仿真引擎。
关闭网页不会停止服务。`sim-web` 终端按 Ctrl+C 关闭控制台；
运行 `docker stop embodied-agent-sim` 停止并自动移除临时仿真容器。

## 当前边界

- 固定桌面场景：`red_block` 与 `box`，只验收积木入盒任务。
- 机械臂采用两连杆逆运动学与预设笛卡尔路径；夹爪抓住物体后，物体位置跟随夹爪。
- 没有接触动力学、摩擦、重力或碰撞检测。`inside_box` 是几何位置判断。
- `verify` 检查仿真状态；尚未接入视觉识别、LeRobot 策略或真实机器人。
- 仿真 Topic 带独立 `/sim/` 前缀，原有 `make ros-demo` 仍是单独的消息接受测试。

## 验证

```bash
make test
make ros-test ros-demo
python3.12 scripts/check_simulation.py  # 两个服务运行时，执行并重置测试场景
```

单元测试覆盖抓取附着、放置后位置验证、错误对象、缺失目标参数与停止后不再运动。
端到端演示需按上述方法启动服务，并分别运行规则计划和 Qwen 计划。
