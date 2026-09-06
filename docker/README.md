# ROS 2 environment

本机已经安装 Docker Desktop，ROS 2 Jazzy 使用 ROS 2 Tooling Working Group 发布的
Ubuntu Noble/ARM64 `ros-base` 镜像。构建项目镜像：

```bash
docker build -t embodied-agent-ros2 -f docker/ros2.Dockerfile .
docker run --rm -it embodied-agent-ros2
```

不构建项目镜像也可以直接进入已下载的 ROS 2 环境：

```bash
docker run --rm -it \
  ghcr.io/ros-tooling/setup-ros-docker/setup-ros-docker-ubuntu-noble-ros-jazzy-ros-base:master
```

Apple Silicon 可以运行该基础镜像；但 USB 串口、相机、GPU 和实时控制通常更适合在 Ubuntu 24.04 机器人主机上运行。

RAI 建议按其官方安装文档加入同一 Ubuntu 环境。LeRobot 的训练/推理可以独立进程运行，通过 ROS 2 topic/service/action 或网络 RPC 与机器人进程通信，降低 Python 与系统依赖冲突。

说明：标准 `ros:jazzy-ros-base` 位于 Docker Hub，但当前网络连接 Docker Hub 超时，因此项目使用内容等价、由 ROS 官方工具组织维护并发布在 GHCR 的开发镜像。
