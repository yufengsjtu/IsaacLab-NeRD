#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/source/isaaclab_neural:${REPO_ROOT}/source/isaaclab:${REPO_ROOT}/source/isaaclab_tasks${PYTHONPATH:+:$PYTHONPATH}"

exec "${REPO_ROOT}/isaaclab.sh" -p -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Lift-Cube-Franka-Dataset-Gen-v0 \
    --env-name Franka-Lift \
    --robot-name Franka-Lift \
    --sample-mode action \
    --initial-states-source env \
    --contact-mode newton_native \
    --num-contacts-per-env 64 \
    --states-frame world \
    "$@"
