# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 helpers used by isaaclab_neural NeRD pipelines."""

from __future__ import annotations

import re
from collections.abc import Sequence

G1_DATASET_COMMAND_LIN_VEL_X = (-1.0, 1.0)
"""Dataset-collection forward/backward command range [m/s]."""

G1_DATASET_COMMAND_LIN_VEL_Y = (-0.5, 0.5)
"""Dataset-collection lateral command range [m/s]."""

G1_DATASET_COMMAND_ANG_VEL_Z = (-1.0, 1.0)
"""Dataset-collection yaw-rate command range [rad/s]."""

G1_DATASET_COMMAND_RESAMPLE_S = (1.0, 1.0)
"""Command resampling period [s], matching 50 control ticks at 50 Hz."""

G1_HAND_JOINT_NAME_PATTERN = re.compile(r".*_(zero|one|two|three|four|five|six)_joint$")
"""Stock ``G1_MINIMAL_CFG`` hand joints (14 DoF), not the 23 locomotion joints."""


def g1_hand_joint_indices(joint_names: Sequence[str]) -> list[int]:
    """Return action/joint indices of stock G1 numbered hand joints.

    Args:
        joint_names: Joint names in action-term or articulation order.

    Returns:
        Indices whose names match :data:`G1_HAND_JOINT_NAME_PATTERN`.
    """
    return [index for index, name in enumerate(joint_names) if G1_HAND_JOINT_NAME_PATTERN.fullmatch(name)]


def apply_g1_rich_velocity_commands(env_cfg) -> None:
    """Sample independent vx/vy/yaw commands and resample them every 1 s.

    Matches the G1 Lab-policy generator: uniform ``[-1, 1] x [-0.5, 0.5] x [-1, 1]``
    with no heading-tracking wrapper and no standing-only fraction.

    Args:
        env_cfg: Manager-based G1 env config that exposes ``commands.base_velocity``.
    """
    command = env_cfg.commands.base_velocity
    command.ranges.lin_vel_x = G1_DATASET_COMMAND_LIN_VEL_X
    command.ranges.lin_vel_y = G1_DATASET_COMMAND_LIN_VEL_Y
    command.ranges.ang_vel_z = G1_DATASET_COMMAND_ANG_VEL_Z
    command.ranges.heading = None
    command.heading_command = False
    command.resampling_time_range = G1_DATASET_COMMAND_RESAMPLE_S
    command.rel_standing_envs = 0.0
