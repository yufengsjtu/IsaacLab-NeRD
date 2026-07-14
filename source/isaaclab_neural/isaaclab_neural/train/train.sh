#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

cd "$(dirname "$0")/../../../.."

PY="./isaaclab.sh -p"
GENERATE_DATASETS="${GENERATE_DATASETS:-true}"
DATASET_DIR="./data/datasets"
LOGDIR="./data/trained_models/Anymal-C-Native"
ENV_NAME="Anymal-C-Native"
ROBOT_NAME="Anymal-C"
GEN_TASK="Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0"
DEPLOY_TASK="Isaac-Velocity-Flat-Anymal-C-v0"
TRAIN_TASK="Isaac-Velocity-Flat-Anymal-C-NeRD-v0"
TRAIN_CFG="./source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_native.yaml"
POLICY_CKPT="./source/isaaclab_neural/isaaclab_neural/generate/Anymal_C/datagen_policies/rsl_rl/Anymal-C-Velocity-Flat/model.pt"

COMMON_GEN_ARGS=(
  --dataset-dir "$DATASET_DIR"
  --env-name "$ENV_NAME"
  --robot-name "$ROBOT_NAME"
  --sample-mode action
  --initial-states-source env
  --contact-mode newton_native
  --num-contacts-per-env 64
  --contact-packing-policy penetration_priority
  --num-envs 1024
  --trajectory-length 400
  --write-chunk-transitions 5000000
  --headless
  --force-overwrite
)

gen() {
  local task="$1" name="$2" transitions="$3" seed="$4"
  shift 4
  $PY -m isaaclab_neural.generate.generate_dataset \
    --task "$task" \
    --dataset-name "$name" \
    --num-transitions "$transitions" \
    --seed "$seed" \
    "${COMMON_GEN_ARGS[@]}" \
    "$@"
}

case "${GENERATE_DATASETS,,}" in
  true | 1 | yes)
    gen "$GEN_TASK" dataset_train.hdf5 20000000 0 \
      --randomize-pd-gains --kp-min 30.0 --kp-max 200.0 --kd-min 0.0 --kd-max 4.0
    gen "$GEN_TASK" dataset_valid.hdf5 1000000 10 \
      --randomize-pd-gains --kp-min 30.0 --kp-max 200.0 --kd-min 0.0 --kd-max 4.0
    gen "$GEN_TASK" dataset_zero_action_valid.hdf5 1000000 20 --zero-actions
    gen "$DEPLOY_TASK" dataset_lstm_actuator_zero_action_valid.hdf5 1000000 30 --zero-actions

    if [[ -f "$POLICY_CKPT" ]]; then
      gen "$DEPLOY_TASK" dataset_lstm_actuator_policy_valid.hdf5 1000000 40 \
        --sample-mode policy --policy-checkpoint "$POLICY_CKPT"
    fi
    ;;
  false | 0 | no)
    echo "[INFO] Skipping dataset generation (GENERATE_DATASETS=$GENERATE_DATASETS)."
    ;;
  *)
    echo "[ERROR] GENERATE_DATASETS must be true or false, got: $GENERATE_DATASETS" >&2
    exit 2
    ;;
esac

$PY -m isaaclab_neural.train.train \
  --task "$TRAIN_TASK" \
  --cfg "$TRAIN_CFG" \
  --logdir "$LOGDIR" \
  --num-envs 1024 \
  --seed 0 \
  --headless \
  --update-dataset-statistics \
  --skip-check-log-override \
  presets=newton_mjwarp
