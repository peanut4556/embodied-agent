# Architecture

```text
User instruction
      |
      v
RAI planner adapter ----> structured Plan[PlanStep]
      |                            |
      |                            v
      |                       SafetyGate
      |                            |
      |                            v
ROS 2 observations -----> LeRobot policy adapter -----> normalized Action
      ^                                                   |
      |                                                   v
      +---------------- robot / simulator <--------- ROS 2 controller
```

## Contracts

- `Planner.create_plan(instruction, observation) -> Plan`
- `Policy.select_action(step, observation) -> Action`
- `Robot.observe() -> Observation`
- `Robot.execute(action)`

`PlanStep` 是语义任务，例如 `pick(red_block)`；LeRobot 的原始输出通常是关节或末端执行器动作。真实项目中，`decode_action` 必须结合机器人动作空间实现，不能直接把模型输出当作安全控制命令。

## Python 环境边界

当前 LeRobot 要求 NumPy 2.x，而 RAI Core 要求 NumPy 1.x，因此两者必须使用独立虚拟环境或容器。跨环境只传输 JSON/ROS message，不共享 Python 对象：

```text
RAI (.venv-rai) -> semantic action JSON -> ROS 2 safety bridge
LeRobot (.venv)  -> normalized policy action -> ROS 2 safety bridge
```

## 真机前必须补充

- 独立于大模型进程的硬件急停
- 关节位置、速度、力矩和工作空间限制
- 通信超时后的安全停止
- 观测时间戳与动作序列号
- 碰撞检测和控制器级限幅
- 每一步的成功/失败检测以及重规划
- 任务与动作日志，便于回放和评测
