OLLAMA ?= ollama
OLLAMA_MODEL ?= qwen3.5:4b

.PHONY: demo rai-demo test deps-check model-pull ollama-check ros-build ros-shell ros-check ros-test ros-demo verify

demo:
	PYTHONPATH=src python3.12 -m embodied_agent "把桌上的红色积木放进盒子"

rai-demo:
	PYTHONPATH=src .venv/bin/python -m embodied_agent --planner rai "把桌上的红色积木放进盒子"

test:
	PYTHONPATH=src python3.12 -m unittest discover -s tests -v

deps-check:
	.venv/bin/python -c 'import importlib.metadata; print("lerobot=" + importlib.metadata.version("lerobot"))'
	.venv-rai/bin/python -c 'import importlib.metadata; print("rai-core=" + importlib.metadata.version("rai-core"))'

model-pull:
	$(OLLAMA) pull $(OLLAMA_MODEL)

ollama-check:
	curl -fsS http://127.0.0.1:11434/api/version
	$(OLLAMA) list | grep -F '$(OLLAMA_MODEL)'

ros-shell:
	docker run --rm -it \
		-v "$(CURDIR):/workspace/embodied-agent-dev" \
		embodied-agent-ros2 \
		bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspace/ros2_ws/install/setup.bash && exec bash'

ros-build:
	docker build -t embodied-agent-ros2 -f docker/ros2.Dockerfile .

ros-check:
	docker run --rm \
		ghcr.io/ros-tooling/setup-ros-docker/setup-ros-docker-ubuntu-noble-ros-jazzy-ros-base:master \
		bash -lc 'source /opt/ros/jazzy/setup.bash && echo ROS_DISTRO=$$ROS_DISTRO && ros2 pkg list | grep -E "^(rclpy|ros2cli|std_msgs)$$"'

ros-test: ros-build
	docker run --rm embodied-agent-ros2 \
		bash -lc 'source /opt/ros/jazzy/setup.bash && cd /workspace/ros2_ws && colcon test --packages-select embodied_agent_ros --event-handlers console_direct+ && colcon test-result --verbose'

ros-demo: ros-build
	docker run --rm embodied-agent-ros2 \
		bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspace/ros2_ws/install/setup.bash && ros2 run embodied_agent_ros loopback'

verify: test deps-check ros-check ros-test ros-demo

.PHONY: sim-ros sim-web
sim-ros: ros-build
	docker run --rm --name embodied-agent-sim -p 127.0.0.1:8766:8766 \
		embodied-agent-ros2 bash -lc 'source /opt/ros/jazzy/setup.bash && source /workspace/ros2_ws/install/setup.bash && ros2 run embodied_agent_ros simulation'

sim-web:
	PYTHONPATH=src .venv/bin/python -m embodied_agent.sim_app

.PHONY: sim-web-background
sim-web-background:
	.venv/bin/python scripts/start_sim_web.py

.PHONY: physics-build physics-ros physics-test
physics-build: ros-build
	docker build -t embodied-agent-physics -f docker/physics.Dockerfile .

physics-ros: physics-build
	docker run --rm --name embodied-agent-sim -p 127.0.0.1:8766:8766 embodied-agent-physics

physics-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/physics -v
