# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


@wp.kernel(enable_backward=False)
def assign_states(
    states: wp.array(dtype=float, ndim=2),
    q_count: int,
    qd_count: int,
    # output
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
):
    """Assign states to joint_q and joint_qd."""

    tid = wp.tid()
    for i in range(q_count):
        joint_q[tid * q_count + i] = states[tid, i]
    for i in range(qd_count):
        joint_qd[tid * qd_count + i] = states[tid, i + q_count]


@wp.kernel(enable_backward=False)
def acquire_states(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    q_count: int,
    qd_count: int,
    # output
    states: wp.array(dtype=float, ndim=2),
):
    """Acquire states from joint_q and joint_qd."""

    tid = wp.tid()
    for i in range(q_count):
        states[tid, i] = joint_q[tid * q_count + i]
    for i in range(qd_count):
        states[tid, q_count + i] = joint_qd[tid * qd_count + i]


def device_to_torch(warp_device):
    """Convert warp device to torch device."""

    return wp.device_to_torch(warp_device)
