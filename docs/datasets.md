# 视觉抓取演示数据集

这一阶段采集 RGB-D 定位和预设 IK 轨迹产生的成功抓放演示，导出为本地
LeRobotDataset v3.0。采集器运行独立的 MuJoCo 场景，不经过 ROS 控制台。
采集完成后使用 LeRobot 逐帧读回，再只用保存的动作回放抓放过程。

## 安装与采集

```bash
.tools/bin/uv pip install --python .venv/bin/python -r requirements/physics.txt -r requirements/dataset.txt
make record-demo DATASET_ROOT=outputs/datasets/my-first-demo
```

需要可用的 OpenGL 渲染上下文。默认从 x=0.28、0.32、0.40 m 三个位置采集，
以 25 Hz 保存 320×240 RGB 图像及状态，每段包含抓取、放置和一秒稳定阶段。
输出目录必须不存在，脚本不会覆盖已有数据。

自定义初始位置与采样率：

```bash
PYTHONPATH=src .venv/bin/python scripts/record_demonstrations.py \
  --output outputs/datasets/another-demo --positions 0.29 0.34 0.38 --fps 25
```

目前仅允许 x∈[0.28, 0.40] m，y=0，积木直立；支持 25 或 50 Hz。
入口强制 Hugging Face 离线模式，缓存默认放在 `outputs/hf-cache`，不上传数据。
`requirements/dataset.txt` 固定使用 LeRobot 0.6.1 的 dataset 依赖。

## 保存了什么

| 字段 | 内容 |
| --- | --- |
| `observation.state` | 肩、肘、腕、左指、右指的 5 个实际关节位置 |
| `observation.velocity` | 相同顺序的实际关节速度 |
| `observation.images.overhead` | 俯视 RGB 图像，240×320×3，保存为图像数据 |
| `action` | 相同顺序的 5 个绝对关节位置控制目标，不是速度或增量 |
| `phase` | 0 抓取、1 放置、2 稳定 |
| `next.done` | 仅每段最后一帧为 true |
| `task` | 中文任务“把桌上的红色积木放进盒子” |

前三个关节的位置单位为 rad，最后两个为 m；速度分别为 rad/s 和 m/s。
使用 MuJoCo 原生关节符号，不能把控制台二维画面的角度直接当作训练状态。
图像和关节状态在同一个物理时刻读取，动作是该时刻计算出的控制目标；
记录发生在积分之前。物理轨迹生成器以 500 Hz 更新，数据以 25/50 Hz 采样，
因此回放时保持每帧动作直到下一帧，会产生一定轨迹误差。验证报告分别记录
机械臂误差（rad）和夹指误差（m），并要求回放实际完成落盒。

`recording.json` 保存关节定义、相机标定、物理版本、场景文件哈希、初始状态、
视觉定位结果及最终成功状态。积木真值坐标仅用于初始化和验收，未加入训练观测。
保存的相机流是 RGB；深度用于专家定位，暂未作为数据集特征导出。

LeRobot 元数据和 Parquet 数据位于 `meta/`、`data/`。图像嵌入 Parquet，
不需要额外下载视频解码器；使用官方库的 `create → add_frame → save_episode → finalize`
流程生成，详见 [LeRobotDataset v3 文档](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3)。

## 验证和预览

```bash
make dataset-check DATASET_ROOT=outputs/datasets/my-first-demo
make dataset-test
```

验证检查全部帧的维度、数值有效性、图像、时间戳、任务、episode 边界、终止标记、
动作范围及阶段顺序。随后在独立世界中只用保存的动作回放，不调用定位器或 IK。
只有全部成功才将 `recording.json` 标记为 `validated` 并写入成功的 `validation.json`。
验证失败会作废旧成功标记。中断采集会保留不完整目录，不能作为通过验证的数据使用。

`preview.gif` 由第一段数据集中保存的图像生成。回放通过说明动作数据能完成当前任务，
不代表已经训练了策略，也不代表学到的策略能泛化。

本次首批样本位于 `outputs/datasets/rgbd-demo-v2`：3 段、每段 261 帧、总计 783 帧，
三段保存动作回放均成功。目录名是本地采集批次名，数据格式仍为 LeRobot v3.0。
`outputs/` 已忽略，不随代码提交到 Git。

## 下一步训练准备

这三个位置只用于验证数据链路，全部在默认 train split 内，不构成可靠训练集或评测集。
下一阶段应扩大初始位置和视觉条件覆盖、按 episode 留出独立评测集，再训练并在未见场景
测试策略。第一版动作空间必须保持这里的五关节顺序、符号、单位和控制频率一致。
