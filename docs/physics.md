# MuJoCo 三维接触物理演示

这一版将二维坐标仿真升级为 MuJoCo 3.3.7 刚体仿真，继续复用 RAI/Qwen、ROS 2
动作/状态 Topic、中文控制台和最终位置验收。

## 实现了什么

- 三个旋转关节和两个夹爪滑动关节，自定义简化机械臂。
- 位置执行器、重力、桌面/托盘/积木碰撞以及双指摩擦抓取。
- 红色积木是带 freejoint 的自由刚体；只有重置时初始化位置，执行期间不修改其 qpos，
  也不使用焊接约束或附着规则将它固定在夹爪上。
- 抓取成功需要双侧指尖接触，并且积木已离开桌面。放置成功需要积木在托盘内、
  已释放且线速度低于 0.03 m/s。动作结束反馈与实际状态一致。
- 三维画面由容器内 MuJoCo/OSMesa 渲染，再通过本机 HTTP 传到浏览器，无外部 CDN。
- 镜像默认启用 `SIM_PERCEPTION=rgbd`：独立俯视相机输出 RGB 和深度，依据红色区域、
  相机标定和已知 5 cm 积木尺寸估计中心位置。`locate` 和 `pick` 均重新拍摄，
  `pick` 使用该估计值生成轨迹，不读取积木真值坐标作为抓取目标。

## 启动

启动 Docker Desktop 和 Ollama，先构建镜像：

```bash
make physics-build
```

首次构建会下载物理引擎和软件渲染依赖。二维版和三维版使用同一端口和容器名称，
如果已有二维仿真运行，请在它没有任务时停止：

```bash
docker stop embodied-agent-sim
```

启动三维仿真和控制台：

```bash
docker run -d --rm --name embodied-agent-sim \
  -p 127.0.0.1:8766:8766 embodied-agent-physics
make sim-web-background
```

打开 <http://127.0.0.1:8765>，默认任务为“把桌上的红色积木放进盒子”。
选择规则计划可快速检查物理执行；选择 Qwen 可验证完整模型规划链路。
既有旧版控制台进程需要先按 `outputs/sim-web.pid` 核对并停止，重新启动后才支持三维图片代理。

## 测试

容器镜像包含运行依赖；如需在宿主机跑物理单元测试，安装独立列出的依赖：

```bash
uv pip install --python .venv/bin/python -r requirements/physics.txt
make physics-test
```

单元测试覆盖：自由落体及桌面支撑、接触抓取与落入托盘、零摩擦抓取失败、
停止轨迹但继续运行物理，以及提前验证失败。两个服务运行时可执行：

```bash
.venv/bin/python scripts/check_simulation.py
```

连同 Qwen 中文规划一起验证，并保存计划、ROS 反馈和最终状态：

```bash
.venv/bin/python scripts/check_simulation.py --rai --report outputs/physics-validation.json
```

该模式为模型规划及执行预留 250 秒，成功后保留落盒画面，便于在控制台查看。

### RGB-D 视觉抓取

```bash
make physics-test
make vision-test
.venv/bin/python scripts/check_simulation.py --rai --expect-perception rgbd --report outputs/vision-validation.json
```

`vision-test` 需要可用的 OpenGL 渲染上下文。它将积木分别放在 x=0.28、0.32、0.40 m，
检查图像估计与真值误差小于 4 mm，并实际完成每个位置的接触抓取和落盒。
真值仅用于测试评分。测试还检查目标消失后不会复用旧坐标，以及偏离机械臂运动平面的
目标会被拒绝。无图形上下文环境可以在镜像内运行：

```bash
docker run --rm -v "$PWD/tests:/tests:ro" embodied-agent-physics \
  /opt/physics/bin/python -m unittest discover -s /tests/vision -v
```

检测结果通过 ROS 状态中的 `detection` 返回，包含来源、估计坐标、像素框和仿真采样时间；
控制台事件会显示坐标。该记录是最近一次定位结果，并非连续跟踪。
目标缺失、多目标、明显遮挡、深度无效或目标不在可抓取平面时会失败，不回退到真值定位。
可显式添加容器参数 `-e SIM_PERCEPTION=truth` 对照旧版真值定位。

投影使用 MuJoCo 相机的内外参和米制深度，坐标约定参见
[MuJoCo 相机文档](https://mujoco.readthedocs.io/en/stable/programming/visualization.html#cameras)。

它检查规则计划、ROS 动作返回、最终物理状态、停止和重置。单条动作有效期为 20 秒，
ROS 状态等待为 22 秒。物理停止指令保持当前关节位置目标；关节仍会有短暂的动态收敛，
重力和碰撞持续生效。

## 当前限制与后续接入

机械臂几何及控制参数用于开发验证，不对应特定商品机械臂。末端轨迹限制在 XZ 平面，
不代表任意三维目标都能抓取。MuJoCo 对接触的模拟不保证与真机一致。
当前视觉模块是受控场景中的颜色与深度几何算法，仅支持已知尺寸、直立且顶面充分可见的
红色积木，并非通用物体识别或 VLA。相机标定和深度来自仿真，尚未加入传感器噪声。
规划器收到的高层物体关系、接触反馈及最终验收仍来自仿真真值，盒子位置仍预先配置。
控制使用逆运动学和预设抓放轨迹，尚未使用 LeRobot 学习策略。
这个环境可以继续用于生成演示轨迹、采集关节/图像数据，
再训练与该机器人关节定义一致的策略。
