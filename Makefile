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

.PHONY: vision-test
vision-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -v

DATASET_ROOT ?= outputs/datasets/rgbd-demo
.PHONY: record-demo dataset-check dataset-test
record-demo:
	PYTHONPATH=src .venv/bin/python scripts/record_demonstrations.py --output "$(DATASET_ROOT)"

dataset-check:
	PYTHONPATH=src .venv/bin/python scripts/record_demonstrations.py --output "$(DATASET_ROOT)" --validate-only

dataset-test:
	PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=outputs/hf-cache .venv/bin/python -m unittest discover -s tests/dataset -v

BC_DATASET ?= outputs/datasets/bc-positions-v1
BC_MODEL ?= outputs/models/context-bc-v1
BC_EVALUATION ?= outputs/evaluations/context-bc-v1
.PHONY: imitation-train imitation-evaluate
imitation-train:
	PYTHONPATH=src .venv/bin/python scripts/train_imitation.py train --dataset "$(BC_DATASET)" --output "$(BC_MODEL)"

imitation-evaluate:
	PYTHONPATH=src .venv/bin/python scripts/train_imitation.py evaluate --dataset "$(BC_DATASET)" --model "$(BC_MODEL)" --output "$(BC_EVALUATION)"

FEEDBACK_OUTPUT ?= outputs/evaluations/feedback-test-v1
.PHONY: feedback-evaluate
feedback-evaluate:
	PYTHONPATH=src .venv/bin/python scripts/evaluate_feedback.py --model "$(BC_MODEL)" --output "$(FEEDBACK_OUTPUT)"

.PHONY: physics-feedback
physics-feedback: physics-build
	test -f "$(BC_MODEL)/policy.npz" && test -f "$(BC_MODEL)/training.json"
	docker run --rm --name embodied-agent-sim -p 127.0.0.1:8766:8766 \
		-v "$(abspath $(BC_MODEL)):/models/context-bc:ro" -e SIM_POLICY_DIR=/models/context-bc \
		embodied-agent-physics

SLIP_OUTPUT ?= outputs/evaluations/slip-test-new
.PHONY: slip-evaluate
slip-evaluate:
	PYTHONPATH=src .venv/bin/python scripts/evaluate_slip.py --model "$(BC_MODEL)" --output "$(SLIP_OUTPUT)"

RECOVERY_DATASET ?= outputs/datasets/recovery-new
.PHONY: recovery-record recovery-check recovery-data-test
recovery-record:
	PYTHONPATH=src .venv/bin/python scripts/record_recovery.py --model "$(BC_MODEL)" --output "$(RECOVERY_DATASET)"

recovery-check:
	PYTHONPATH=src .venv/bin/python scripts/record_recovery.py --output "$(RECOVERY_DATASET)" --validate-only

recovery-data-test:
	PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=outputs/hf-cache RECOVERY_TEST_MODEL="$(BC_MODEL)" .venv/bin/python -m unittest discover -s tests/dataset -p test_recovery_data.py -v

TEMPORAL_DATASET ?= outputs/datasets/recovery-v2
TEMPORAL_INDEX ?= outputs/datasets/recovery-v2-temporal-v1.json
TEMPORAL_SPLIT ?= config/temporal-curation-split.json
.PHONY: temporal-prepare temporal-test
temporal-prepare:
	PYTHONPATH=src .venv/bin/python scripts/prepare_temporal.py --dataset "$(TEMPORAL_DATASET)" --output "$(TEMPORAL_INDEX)" --split "$(TEMPORAL_SPLIT)"

temporal-test:
	PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=outputs/hf-cache .venv/bin/python -m unittest discover -s tests/dataset -p test_temporal_data.py -v

REACTIVE_DATASET ?= outputs/datasets/reactive-development-v2
REACTIVE_INDEX ?= outputs/datasets/reactive-development-v2-temporal.json
REACTIVE_MODEL ?= outputs/models/reactive-bc-v1
REACTIVE_EVALUATION ?= outputs/evaluations/reactive-bc-v1
REACTIVE_EXPERIMENT ?= config/reactive-experiment.json
.PHONY: reactive-record reactive-prepare reactive-train reactive-evaluate reactive-test
reactive-record:
	PYTHONPATH=src .venv/bin/python scripts/record_recovery.py --output "$(REACTIVE_DATASET)" --model "$(BC_MODEL)" --scenarios "$(REACTIVE_EXPERIMENT)" --group development

reactive-prepare:
	PYTHONPATH=src .venv/bin/python scripts/train_reactive.py prepare --dataset "$(REACTIVE_DATASET)" --experiment "$(REACTIVE_EXPERIMENT)" --output "$(REACTIVE_INDEX)"

reactive-train:
	PYTHONPATH=src .venv/bin/python scripts/train_reactive.py train --index "$(REACTIVE_INDEX)" --experiment "$(REACTIVE_EXPERIMENT)" --output "$(REACTIVE_MODEL)"

reactive-evaluate:
	PYTHONPATH=src .venv/bin/python scripts/train_reactive.py evaluate --model "$(REACTIVE_MODEL)" --baseline "$(BC_MODEL)" --output "$(REACTIVE_EVALUATION)"

reactive-test:
	PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=outputs/hf-cache .venv/bin/python -m unittest discover -s tests/dataset -p test_reactive.py -v

MEMORY_DATASET ?= outputs/datasets/reactive-development-v2
MEMORY_RUN ?= outputs/models/memory-bc-v2
MEMORY_EXPERIMENT ?= config/memory-extended-experiment.json
MEMORY_SELECTION ?= outputs/evaluations/memory-bc-v2-development
MEMORY_EVALUATION ?= outputs/evaluations/memory-bc-v2-test
.PHONY: memory-train memory-select memory-evaluate memory-test
memory-train:
	PYTHONPATH=src .venv/bin/python scripts/train_memory.py train --dataset "$(MEMORY_DATASET)" --experiment "$(MEMORY_EXPERIMENT)" --output "$(MEMORY_RUN)"

memory-select:
	PYTHONPATH=src .venv/bin/python scripts/train_memory.py select --run "$(MEMORY_RUN)" --output "$(MEMORY_SELECTION)"

memory-evaluate:
	PYTHONPATH=src .venv/bin/python scripts/train_memory.py evaluate --run "$(MEMORY_RUN)" --selection "$(MEMORY_SELECTION)/selection.json" --baseline "$(BC_MODEL)" --output "$(MEMORY_EVALUATION)"

memory-test:
	PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=outputs/hf-cache .venv/bin/python -m unittest discover -s tests/dataset -p test_memory_policy.py -v

CORRECTION_DATASET ?= outputs/datasets/corrections-v2
CORRECTION_MODEL ?= outputs/models/memory-bc-v2/epoch-2000
.PHONY: correction-record correction-check correction-test
correction-record:
	PYTHONPATH=src .venv/bin/python scripts/record_corrections.py --model "$(CORRECTION_MODEL)" --output "$(CORRECTION_DATASET)"

correction-check:
	PYTHONPATH=src .venv/bin/python scripts/record_corrections.py --output "$(CORRECTION_DATASET)" --validate-only

correction-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/physics -p test_grasp_quality.py -v
	PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=outputs/hf-cache CORRECTION_TEST_DATASET="$(CORRECTION_DATASET)" .venv/bin/python -m unittest discover -s tests/dataset -p test_correction_data.py -v

.PHONY: correction-expand correction-finetune correction-select correction-finetune-test
correction-expand:
	PYTHONPATH=src .venv/bin/python scripts/record_corrections.py --output outputs/datasets/corrections-expanded-v1 --scenarios config/correction-expanded-scenarios.json

correction-finetune:
	PYTHONPATH=src .venv/bin/python scripts/finetune_corrections.py

correction-select:
	PYTHONPATH=src .venv/bin/python scripts/train_memory.py select --run outputs/models/memory-correction-v1 --output outputs/evaluations/memory-correction-v1-development

correction-finetune-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/dataset -p test_correction_finetune.py -v

.PHONY: correction-diagnose correction-early-record correction-early-train correction-early-select diagnostics-test
correction-diagnose:
	PYTHONPATH=src .venv/bin/python scripts/diagnose_correction_policy.py

correction-early-record:
	PYTHONPATH=src .venv/bin/python scripts/record_corrections.py --model outputs/models/memory-correction-v1/epoch-600 --scenarios config/correction-early-scenarios.json --output outputs/datasets/corrections-early-v1

correction-early-train:
	PYTHONPATH=src .venv/bin/python scripts/finetune_corrections.py --pretrained outputs/models/memory-correction-v1/epoch-600 --corrections outputs/datasets/corrections-early-v1 --experiment config/correction-early-experiment.json --output outputs/models/memory-correction-v2

correction-early-select:
	PYTHONPATH=src .venv/bin/python scripts/train_memory.py select --run outputs/models/memory-correction-v2 --output outputs/evaluations/memory-correction-v2-development

diagnostics-test:
	PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/physics -p test_policy_diagnostics.py -v

.PHONY: action-chain-audit action-chain-test
action-chain-audit:
	PYTHONPATH=src:. .venv/bin/python scripts/audit_action_chain.py

action-chain-test:
	PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/physics -p test_action_chain.py -v
