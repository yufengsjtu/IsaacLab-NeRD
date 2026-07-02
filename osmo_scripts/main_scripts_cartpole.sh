#!/bin/bash
# Generate Cartpole fixed-ground datasets and train IsaacLab-NeRD in one OSMO task.

set -euxo pipefail

PROJECT_ROOT="$HOME/code/IsaacLab-NeRD"
OUTPUT_ROOT="${OUTPUT_LOCAL_PATH:-/tmp/runs/output}"
DATASET_DIR="${DATASET_DIR:-./data/datasets}"
TRAIN_TRANSITIONS="${TRAIN_TRANSITIONS:-1000000}"
VALID_TRANSITIONS="${VALID_TRANSITIONS:-100000}"
NUM_ENVS="${NUM_ENVS:-256}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-100}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"

python3 -m tensorboard.main --logdir "$OUTPUT_ROOT" --host 0.0.0.0 --port "$TENSORBOARD_PORT" \
    > "$OUTPUT_ROOT/tensorboard.log" 2>&1 &
echo "$!" > "$OUTPUT_ROOT/tensorboard.pid"
echo "TensorBoard started on port $TENSORBOARD_PORT with logdir $OUTPUT_ROOT"

python3 -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Cartpole-v0 \
    --dataset-dir "$DATASET_DIR" \
    --dataset-name dataset_train.hdf5 \
    --env-name Cartpole \
    --robot-name Cartpole \
    --sample-mode joint_f \
    --initial-states-source sample \
    --contact-mode fixed_ground \
    --states-frame world \
    --num-envs "$NUM_ENVS" \
    --num-transitions "$TRAIN_TRANSITIONS" \
    --trajectory-length "$TRAJECTORY_LENGTH" \
    --seed 0 \
    --headless \
    --force-overwrite

python3 -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Cartpole-v0 \
    --dataset-dir "$DATASET_DIR" \
    --dataset-name dataset_valid.hdf5 \
    --env-name Cartpole \
    --robot-name Cartpole \
    --sample-mode joint_f \
    --initial-states-source sample \
    --contact-mode fixed_ground \
    --states-frame world \
    --num-envs "$NUM_ENVS" \
    --num-transitions "$VALID_TRANSITIONS" \
    --trajectory-length "$TRAJECTORY_LENGTH" \
    --seed 1 \
    --headless \
    --force-overwrite

python3 -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Cartpole-v0 \
    --dataset-dir "$DATASET_DIR" \
    --dataset-name dataset_passive_valid.hdf5 \
    --env-name Cartpole \
    --robot-name Cartpole \
    --sample-mode joint_f \
    --initial-states-source sample \
    --contact-mode fixed_ground \
    --states-frame world \
    --zero-actions \
    --num-envs "$NUM_ENVS" \
    --num-transitions "$VALID_TRANSITIONS" \
    --trajectory-length "$TRAJECTORY_LENGTH" \
    --seed 2 \
    --headless \
    --force-overwrite

python3 -m isaaclab_neural.train.train \
    --task Isaac-Cartpole-NeRD-v0 \
    --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Cartpole/transformer.yaml \
    --logdir "$OUTPUT_ROOT/Cartpole" \
    --num-envs "$NUM_ENVS" \
    --seed 0 \
    --headless \
    --skip-check-log-override \