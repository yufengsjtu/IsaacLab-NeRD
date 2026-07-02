#!/bin/bash
# Generate Anymal-C suite datasets and train IsaacLab-NeRD in one OSMO task.

set -euxo pipefail

PROJECT_ROOT="$HOME/code/IsaacLab-NeRD"
OUTPUT_ROOT="${OUTPUT_LOCAL_PATH:-/tmp/runs/output}"
DATASET_DIR="${DATASET_DIR:-./data/datasets}"
TRAIN_TRANSITIONS="${TRAIN_TRANSITIONS:-20000000}"
VALID_TRANSITIONS="${VALID_TRANSITIONS:-1000000}"
NUM_ENVS="${NUM_ENVS:-1024}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-400}"
WRITE_CHUNK_TRANSITIONS="${WRITE_CHUNK_TRANSITIONS:-10000000}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-source/isaaclab_neural/isaaclab_neural/generate/Anymal_C/datagen_policies/rsl_rl/Anymal-C-Velocity-Flat/model.pt}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"

python3 -m tensorboard.main --logdir "$OUTPUT_ROOT" --host 0.0.0.0 --port "$TENSORBOARD_PORT" \
    > "$OUTPUT_ROOT/tensorboard.log" 2>&1 &
echo "$!" > "$OUTPUT_ROOT/tensorboard.pid"
echo "TensorBoard started on port $TENSORBOARD_PORT with logdir $OUTPUT_ROOT"

python3 -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
    --dataset-dir "$DATASET_DIR" \
    --dataset-name dataset_train.hdf5 \
    --env-name Anymal-C \
    --robot-name Anymal-C \
    --sample-mode action \
    --initial-states-source env \
    --contact-mode fixed_ground \
    --randomize-pd-gains \
    --kp-min 30.0 \
    --kp-max 200.0 \
    --kd-min 0.0 \
    --kd-max 4.0 \
    --num-envs "$NUM_ENVS" \
    --num-transitions "$TRAIN_TRANSITIONS" \
    --write-chunk-transitions "$WRITE_CHUNK_TRANSITIONS" \
    --trajectory-length "$TRAJECTORY_LENGTH" \
    --seed 0 \
    --headless \
    --force-overwrite

python3 -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
    --dataset-dir "$DATASET_DIR" \
    --dataset-name dataset_valid.hdf5 \
    --env-name Anymal-C \
    --robot-name Anymal-C \
    --sample-mode action \
    --initial-states-source env \
    --contact-mode fixed_ground \
    --randomize-pd-gains \
    --kp-min 30.0 \
    --kp-max 200.0 \
    --kd-min 0.0 \
    --kd-max 4.0 \
    --num-envs "$NUM_ENVS" \
    --num-transitions "$VALID_TRANSITIONS" \
    --write-chunk-transitions "$WRITE_CHUNK_TRANSITIONS" \
    --trajectory-length "$TRAJECTORY_LENGTH" \
    --seed 10 \
    --headless \
    --force-overwrite

# Zero-action validation on the same Velocity-Flat task. For Anymal's
# JointPositionAction this means "hold the default standing pose".
python3 -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
    --dataset-dir "$DATASET_DIR" \
    --dataset-name dataset_zero_action_valid.hdf5 \
    --env-name Anymal-C \
    --robot-name Anymal-C \
    --sample-mode action \
    --initial-states-source env \
    --contact-mode fixed_ground \
    --zero-actions \
    --num-envs "$NUM_ENVS" \
    --num-transitions "$VALID_TRANSITIONS" \
    --write-chunk-transitions "$WRITE_CHUNK_TRANSITIONS" \
    --trajectory-length "$TRAJECTORY_LENGTH" \
    --seed 20 \
    --headless \
    --force-overwrite

# Same zero-action validation without randomized PD gains. This mirrors the
# deployment-time LSTM-actuator validation set from the original suite.
python3 -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Velocity-Flat-Anymal-C-v0 \
    --dataset-dir "$DATASET_DIR" \
    --dataset-name dataset_lstm_actuator_zero_action_valid.hdf5 \
    --env-name Anymal-C \
    --robot-name Anymal-C \
    --sample-mode action \
    --initial-states-source env \
    --contact-mode fixed_ground \
    --zero-actions \
    --num-envs "$NUM_ENVS" \
    --num-transitions "$VALID_TRANSITIONS" \
    --write-chunk-transitions "$WRITE_CHUNK_TRANSITIONS" \
    --trajectory-length "$TRAJECTORY_LENGTH" \
    --seed 30 \
    --headless \
    --force-overwrite

# Policy-driven validation on the deployment-time actuator distribution. The
# checkpoint is not committed with this repo, so generate it only when mounted.
if [[ -f "$POLICY_CHECKPOINT" ]]; then
    python3 -m isaaclab_neural.generate.generate_dataset \
        --task Isaac-Velocity-Flat-Anymal-C-v0 \
        --dataset-dir "$DATASET_DIR" \
        --dataset-name dataset_lstm_actuator_policy_valid.hdf5 \
        --env-name Anymal-C \
        --robot-name Anymal-C \
        --sample-mode policy \
        --policy-checkpoint "$POLICY_CHECKPOINT" \
        --initial-states-source env \
        --contact-mode fixed_ground \
        --num-envs "$NUM_ENVS" \
        --num-transitions "$VALID_TRANSITIONS" \
        --write-chunk-transitions "$WRITE_CHUNK_TRANSITIONS" \
        --trajectory-length "$TRAJECTORY_LENGTH" \
        --seed 40 \
        --headless \
        --force-overwrite
else
    echo "Skipping policy validation dataset; checkpoint not found: $POLICY_CHECKPOINT"
fi

python3 -m isaaclab_neural.train.train \
    --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
    --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer.yaml \
    --logdir "$OUTPUT_ROOT/Anymal-C" \
    --num-envs "$NUM_ENVS" \
    --seed 0 \
    --headless \
    --skip-check-log-override \
    presets=newton_mjwarp

