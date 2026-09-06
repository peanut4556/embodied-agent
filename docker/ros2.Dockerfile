FROM ghcr.io/ros-tooling/setup-ros-docker/setup-ros-docker-ubuntu-noble-ros-jazzy-ros-base:master

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace
COPY src /workspace/core
ENV PYTHONPATH=/workspace/core

COPY ros2_ws/src /workspace/ros2_ws/src
RUN source /opt/ros/jazzy/setup.bash \
    && cd /workspace/ros2_ws \
    && colcon build --symlink-install

CMD ["bash"]
