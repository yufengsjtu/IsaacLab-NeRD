#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

exec "${REPO_ROOT}/isaaclab.sh" -p -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Cartpole-v0 \
    --env-name Cartpole \
    --robot-name Cartpole \
    --sample-mode joint_f \
    --initial-states-source sample \
    --contact-mode fixed_ground \
    "$@"
