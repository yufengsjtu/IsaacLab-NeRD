#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
# Conda's editable install points at the main checkout, which does not register
# Isaac-Reach-Franka-Dataset-Gen-v0. Prefer this worktree's packages.
export PYTHONPATH="${REPO_ROOT}/source/isaaclab_neural:${REPO_ROOT}/source/isaaclab:${REPO_ROOT}/source/isaaclab_tasks:${REPO_ROOT}/source/isaaclab_assets:${REPO_ROOT}/source/isaaclab_newton:${REPO_ROOT}/source/isaaclab_rl${PYTHONPATH:+:$PYTHONPATH}"

exec uv run --project "${REPO_ROOT}" --no-sync python -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Reach-Franka-Dataset-Gen-v0 \
    --dataset-dir "${REPO_ROOT}/data/datasets/menagerie-60hz" \
    --env-name Franka-Reach \
    --robot-name Franka \
    --sample-mode action \
    --initial-states-source env \
    --action-hold-frames 20 \
    --trajectory-length 200 \
    --contact-mode newton_native \
    --num-contacts-per-env 0 \
    --states-frame world \
    "$@"
