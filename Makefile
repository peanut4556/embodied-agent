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

.PHONY: input-feedback-audit input-feedback-test
input-feedback-audit:
	PYTHONPATH=src:. .venv/bin/python scripts/audit_input_feedback.py

input-feedback-test:
	PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/physics -p test_input_feedback.py -v

.PHONY: joint-sensitivity-audit joint-sensitivity-test
joint-sensitivity-audit:
	PYTHONPATH=src:. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=outputs/hf-cache .venv/bin/python scripts/audit_joint_sensitivity.py

joint-sensitivity-test:
	PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/physics -p test_joint_sensitivity.py -v

.PHONY: joint-training-control joint-training-augmented joint-training-evaluate joint-augmentation-test
joint-training-control:
	PYTHONPATH=src .venv/bin/python scripts/finetune_corrections.py --pretrained outputs/models/memory-correction-v1/epoch-600 --corrections outputs/datasets/corrections-early-v1 --experiment config/joint-training-control.json --output outputs/models/joint-training-control

joint-training-augmented:
	PYTHONPATH=src .venv/bin/python scripts/finetune_corrections.py --pretrained outputs/models/memory-correction-v1/epoch-600 --corrections outputs/datasets/corrections-early-v1 --experiment config/joint-training-augmented.json --output outputs/models/joint-training-augmented

joint-training-evaluate:
	PYTHONPATH=src:. .venv/bin/python scripts/evaluate_joint_training.py

joint-augmentation-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/dataset -p test_joint_augmentation.py -v

.PHONY: joint-seeds-train joint-seeds-evaluate joint-replication-test
joint-seeds-train:
	PYTHONPATH=src:. .venv/bin/python scripts/replicate_joint_training.py train

joint-seeds-evaluate:
	PYTHONPATH=src:. .venv/bin/python scripts/replicate_joint_training.py evaluate

joint-replication-test:
	PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/dataset -p test_joint_replication.py -v

.PHONY: state-targets-audit state-targets-test
state-targets-audit:
	PYTHONPATH=src:. .venv/bin/python scripts/audit_state_targets.py

state-targets-test:
	PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/physics -p test_state_targets.py -v

.PHONY: state-distill-targets state-distill-control state-distill-train state-distill-evaluate local-targets-test
state-distill-targets:
	PYTHONPATH=src:. .venv/bin/python -c 'from scripts.audit_state_targets import audit; audit("outputs/datasets/state-distill-targets-v1", ("train",))'

state-distill-control:
	PYTHONPATH=src .venv/bin/python scripts/finetune_corrections.py --pretrained outputs/models/joint-training-control/epoch-300 --corrections outputs/datasets/corrections-early-v1 --experiment config/state-distill-control.json --output outputs/models/state-distill-control

state-distill-train:
	PYTHONPATH=src .venv/bin/python scripts/finetune_corrections.py --pretrained outputs/models/joint-training-control/epoch-300 --corrections outputs/datasets/corrections-early-v1 --experiment config/state-distill-distilled.json --output outputs/models/state-distill-distilled

state-distill-evaluate:
	PYTHONPATH=src:. .venv/bin/python scripts/evaluate_state_distillation.py

local-targets-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/dataset -p test_local_targets.py -v

.PHONY: correction-onpolicy-record multistep-targets-audit multistep-targets-test
correction-onpolicy-record:
	PYTHONPATH=src .venv/bin/python scripts/record_corrections.py --model outputs/models/state-distill-distilled/epoch-300 --scenarios config/correction-onpolicy-scenarios.json --output outputs/datasets/corrections-onpolicy-v1

multistep-targets-audit:
	PYTHONPATH=src:. .venv/bin/python scripts/audit_multistep_targets.py

multistep-targets-test:
	PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/physics -p test_multistep_targets.py -v

.PHONY: completion-targets completion-train completion-evaluate future-targets-test
completion-targets:
	PYTHONPATH=src .venv/bin/python scripts/prepare_completion_targets.py

completion-train:
	PYTHONPATH=src .venv/bin/python scripts/finetune_corrections.py --pretrained outputs/models/state-distill-distilled/epoch-300 --corrections outputs/datasets/corrections-onpolicy-v1 --experiment config/completion-bc.json --output outputs/models/completion-bc
	PYTHONPATH=src .venv/bin/python scripts/finetune_corrections.py --pretrained outputs/models/state-distill-distilled/epoch-300 --corrections outputs/datasets/corrections-onpolicy-v1 --experiment config/completion-unmasked.json --output outputs/models/completion-unmasked
	PYTHONPATH=src .venv/bin/python scripts/finetune_corrections.py --pretrained outputs/models/state-distill-distilled/epoch-300 --corrections outputs/datasets/corrections-onpolicy-v1 --experiment config/completion-masked.json --output outputs/models/completion-masked

completion-evaluate:
	PYTHONPATH=src:. .venv/bin/python scripts/evaluate_completion_mask.py

future-targets-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/dataset -p test_future_targets.py -v

.PHONY: success-replay-prepare success-replay-train success-replay-evaluate success-replay-test
success-replay-prepare:
	PYTHONPATH=src .venv/bin/python scripts/prepare_success_replay.py

success-replay-train:
	PYTHONPATH=src .venv/bin/python scripts/finetune_corrections.py --pretrained outputs/models/state-distill-distilled/epoch-300 --corrections outputs/datasets/corrections-onpolicy-v1 --experiment config/success-replay-control.json --output outputs/models/success-replay-control
	PYTHONPATH=src .venv/bin/python scripts/finetune_corrections.py --pretrained outputs/models/state-distill-distilled/epoch-300 --corrections outputs/datasets/corrections-onpolicy-v1 --experiment config/success-replay-replay.json --output outputs/models/success-replay-replay

success-replay-evaluate:
	PYTHONPATH=src .venv/bin/python scripts/evaluate_success_replay.py

success-replay-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/dataset -p test_success_replay.py -v

.PHONY: replay-divergence-audit trajectory-divergence-test
replay-divergence-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_replay_divergence.py
	PYTHONPATH=src .venv/bin/python scripts/summarize_replay_divergence.py

trajectory-divergence-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/physics -p test_trajectory_divergence.py -v

.PHONY: contact-recovery-record contact-recovery-test
contact-recovery-record:
	PYTHONPATH=src .venv/bin/python scripts/collect_contact_recovery.py

contact-recovery-test:
	PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/physics -p test_contact_recovery.py -v

.PHONY: contact-stable-record contact-stable-test
contact-stable-record:
	PYTHONPATH=src:. .venv/bin/python -c 'from scripts.collect_contact_recovery import main; main("config/correction-contact-stable-scenarios.json", "outputs/datasets/corrections-contact-stable-v1", "docs/evaluations/contact-stable-v1.json")'

contact-stable-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/physics -p test_stable_contact_expert.py -v

.PHONY: contact-stable-compare
contact-stable-compare:
	PYTHONPATH=src .venv/bin/python scripts/compare_contact_experts.py

.PHONY: contact-vision-audit contact-vision-summary perception-diagnostics-test
contact-vision-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_contact_vision.py
	PYTHONPATH=src .venv/bin/python scripts/summarize_contact_vision.py

contact-vision-summary:
	PYTHONPATH=src .venv/bin/python scripts/summarize_contact_vision.py

perception-diagnostics-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_perception_diagnostics.py -v

.PHONY: contact-oriented-record contact-oriented-compare
contact-oriented-record:
	PYTHONPATH=src:. .venv/bin/python -c 'from scripts.collect_contact_recovery import main; main("config/correction-contact-oriented-scenarios.json", "outputs/datasets/corrections-contact-oriented-v1", "docs/evaluations/contact-oriented-v1.json")'

contact-oriented-compare:
	PYTHONPATH=src .venv/bin/python scripts/compare_oriented_recovery.py

.PHONY: bounded-wait-audit
bounded-wait-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_bounded_wait.py

.PHONY: support-reach-audit support-reach-test
support-reach-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_support_reach.py

support-reach-test:
	PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests/physics -p test_support_reach.py -v

.PHONY: visible-geometry-audit visible-geometry-test
visible-geometry-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_visible_geometry.py

visible-geometry-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_visible_geometry.py -v

.PHONY: conservative-geometry-audit conservative-geometry-test
conservative-geometry-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_conservative_geometry.py

conservative-geometry-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_conservative_geometry.py -v

.PHONY: geometry-error-audit geometry-error-test
geometry-error-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_geometry_error.py

geometry-error-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_geometry_error.py -v

.PHONY: rgbd-filter-audit rgbd-filter-test
rgbd-filter-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_rgbd_filter.py

rgbd-filter-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_rgbd_filter.py -v

.PHONY: envelope-stress-audit filtered-envelope-test
envelope-stress-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_envelope_stress.py

filtered-envelope-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_filtered_envelope.py -v

.PHONY: gripper-sweep-audit gripper-sweep-test
gripper-sweep-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_gripper_sweep.py

gripper-sweep-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/physics -p test_gripper_sweep.py -v

.PHONY: gripper-sweep-refined-audit
gripper-sweep-refined-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_gripper_sweep.py --refine

.PHONY: overlap-attribution-audit box-overlap-test
overlap-attribution-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_overlap_attribution.py

box-overlap-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/physics -p test_box_overlap.py -v

.PHONY: directional-envelope-audit directional-envelope-test
directional-envelope-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_directional_envelope.py

directional-envelope-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_directional_envelope.py -v

.PHONY: plane-groups-audit plane-groups-test
plane-groups-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_plane_groups.py

plane-groups-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_plane_groups.py -v

.PHONY: plane-stability-audit plane-stability-test
plane-stability-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_plane_stability.py

plane-stability-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_plane_stability.py -v

.PHONY: plane-holdout-audit plane-holdout-test
plane-holdout-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_plane_holdout.py

plane-holdout-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_plane_holdout.py -v

.PHONY: depth-ramp-audit depth-ramp-test
depth-ramp-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_depth_ramp.py

depth-ramp-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_depth_ramp.py -v

.PHONY: cube-consistency-audit cube-consistency-test
cube-consistency-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_cube_consistency.py

cube-consistency-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_cube_consistency.py -v

.PHONY: reference-plane-audit reference-plane-test
reference-plane-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_reference_plane.py

reference-plane-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_reference_plane.py -v

.PHONY: reference-mismatch-audit reference-mismatch-test
reference-mismatch-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_reference_mismatch.py

reference-mismatch-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_reference_mismatch.py -v

.PHONY: multiplane-reference-audit multiplane-reference-test
multiplane-reference-audit:
	PYTHONPATH=src .venv/bin/python scripts/audit_multiplane_reference.py

multiplane-reference-test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/vision -p test_multiplane_reference.py -v
