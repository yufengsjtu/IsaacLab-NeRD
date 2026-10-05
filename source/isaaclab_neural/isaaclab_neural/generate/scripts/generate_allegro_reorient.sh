#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/source/isaaclab_neural:${REPO_ROOT}/source/isaaclab_newton:${REPO_ROOT}/source/isaaclab:${REPO_ROOT}/source/isaaclab_tasks:${REPO_ROOT}/source/isaaclab_assets:${REPO_ROOT}/source/isaaclab_rl${PYTHONPATH:+:$PYTHONPATH}"

exec uv run --no-sync --project "${REPO_ROOT}" python -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Reorient-Cube-Allegro-Dataset-Gen-v0 \
    --env-name Allegro-Reorient \
    --robot-name Allegro-Reorient \
    --dataset-dir "${REPO_ROOT}/data/datasets/allegro-reorient-newton-contacts-v1" \
    --initial-states-source env \
    --trajectory-length 600 \
    --contact-mode newton_native \
    --contact-representation contact_tokens \
    --contact-packing-policy body_round_robin_pair_atomic \
    --num-contacts-per-env 128 \
    --max-contact-tokens 128 \
    --states-frame body \
    "$@"
