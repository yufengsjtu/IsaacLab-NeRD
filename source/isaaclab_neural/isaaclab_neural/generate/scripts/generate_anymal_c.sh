#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

exec "${REPO_ROOT}/isaaclab.sh" -p -m isaaclab_neural.generate.generate_dataset \
    --task Isaac-Velocity-Flat-Anymal-C-v0 \
    --env-name Anymal-C \
    --robot-name Anymal-C \
    --sample-mode action \
    --initial-states-source env \
    --contact-mode fixed_ground \
    "$@"
