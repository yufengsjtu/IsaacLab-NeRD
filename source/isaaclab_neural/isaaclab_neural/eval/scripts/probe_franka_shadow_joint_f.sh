#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/source/isaaclab_neural:${REPO_ROOT}/source/isaaclab_newton:${REPO_ROOT}/source/isaaclab:${REPO_ROOT}/source/isaaclab_tasks${PYTHONPATH:+:$PYTHONPATH}"

POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-${REPO_ROOT}/pretrained/control_policy/franka_reach/model_999.pt}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/shadow_joint_f_probe}"
NUM_ENVS="${NUM_ENVS:-8}"
NUM_STEPS="${NUM_STEPS:-240}"
SEED="${SEED:-0}"

mkdir -p "${OUT_DIR}"

"${REPO_ROOT}/isaaclab.sh" -p -m isaaclab_neural.eval.probe_franka_shadow_joint_f \
    --policy-checkpoint "${POLICY_CHECKPOINT}" \
    --num-envs "${NUM_ENVS}" \
    --num-steps "${NUM_STEPS}" \
    --seed "${SEED}" \
    --visualizer none \
    --metrics-json "${OUT_DIR}/probe.json" \
    presets=newton_mjwarp
