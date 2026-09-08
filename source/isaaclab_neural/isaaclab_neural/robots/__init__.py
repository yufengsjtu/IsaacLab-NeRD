# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot articulation configs for isaaclab_neural."""

from .g1 import G1_HAND_JOINT_NAME_PATTERN, apply_g1_rich_velocity_commands, g1_hand_joint_indices

__all__ = [
    "G1_HAND_JOINT_NAME_PATTERN",
    "apply_g1_rich_velocity_commands",
    "g1_hand_joint_indices",
]
