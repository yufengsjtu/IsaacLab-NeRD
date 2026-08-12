# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp
from newton import JointType


@wp.kernel
def determine_angular_dofs(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    # outputs
    joint_q_end: wp.array(dtype=int),
    is_angular: wp.array(dtype=bool),
    is_continuous: wp.array(dtype=bool),
):
    joint_id = wp.tid()
    q_start = joint_q_start[joint_id]
    limit_start = joint_qd_start[joint_id]  # NOTE: joint limit is in joint_dof_count shape
    type = joint_type[joint_id]
    # Newton closes joint_q_start with a sentinel (= joint_coord_count) at
    # index joint_count, so joint_q_start[joint_id + 1] is always in-bounds
    # and equals the q-end index for this joint regardless of joint type.
    joint_q_end[joint_id] = joint_q_start[joint_id + 1]
    if type == JointType.FREE or type == JointType.DISTANCE or type == JointType.BALL:
        for i in range(q_start, joint_q_end[joint_id]):
            # quaternions do not count as angular dofs
            is_angular[i] = False
    elif type == JointType.PRISMATIC:
        is_angular[q_start] = False
    elif type == JointType.REVOLUTE:
        is_angular[q_start] = True
        lower = joint_limit_lower[limit_start]
        upper = joint_limit_upper[limit_start]
        if upper - lower > 2.0 * wp.pi:
            is_continuous[q_start] = True
    else:  # FIXED, D6, etc.
        for i in range(q_start, joint_q_end[joint_id]):
            is_angular[i] = False
            is_continuous[i] = False
