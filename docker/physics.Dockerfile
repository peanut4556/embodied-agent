FROM embodied-agent-ros2
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 python3-venv \
    && rm -rf /var/lib/apt/lists/*
COPY requirements/physics.txt /tmp/physics.txt
RUN python3 -m venv --system-site-packages /opt/physics \
    && /opt/physics/bin/pip install --no-cache-dir -r /tmp/physics.txt
COPY src /workspace/core
COPY ros2_ws/src /workspace/ros2_ws/src
ENV MUJOCO_GL=osmesa
ENV SIM_ENGINE=mujoco
CMD ["bash", "-lc", "source /opt/ros/jazzy/setup.bash && source /workspace/ros2_ws/install/setup.bash && /opt/physics/bin/python -m embodied_agent_ros.simulation_server"]
