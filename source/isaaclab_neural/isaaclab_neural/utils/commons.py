# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared constants for isaaclab_neural."""

import numpy as np

DATASET_MODES = ("transition", "trajectory")
"""Supported NeRD dataset storage modes."""

CARTPOLE_SMALL_RANGE = False
"""Whether to use the original narrow Cartpole sampling range."""

_FRANKA_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0])
_FRANKA_Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04])
_FRANKA_QD_LIM = np.full(9, 10.0)
_FRANKA_F_LIM = np.array([100.0, 100.0, 100.0, 100.0, 20.0, 20.0, 20.0, 0.0, 0.0])
# Core ``Isaac-Lift-Franka`` frame: base rotated 180 deg about z, table top at z = 0.255 [m].
_FRANKA_LIFT_CUBE_Q_MIN = np.array([-0.9, -0.5, 0.25, -1.0, -1.0, -1.0, -1.0])
_FRANKA_LIFT_CUBE_Q_MAX = np.array([-0.2, 0.5, 0.9, 1.0, 1.0, 1.0, 1.0])
_FRANKA_LIFT_CUBE_QD_LIM = np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
_FRANKA_LIFT_CUBE_F_LIM = np.zeros(6)

JOINT_Q_MIN = {
    "Cartpole": np.array([-1.0, -np.pi]) if CARTPOLE_SMALL_RANGE else np.array([-3.5, -np.pi]),
    "Anymal-C": np.array(
        [
            -10.0,
            0.8,
            -10.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -0.49,
            -np.pi,
            -np.pi,
            -0.72,
            -np.pi,
            -np.pi,
            -0.49,
            -np.pi,
            -np.pi,
            -0.72,
            -np.pi,
            -np.pi,
        ]
    ),
    "Franka": _FRANKA_Q_MIN,
    "Franka-Lift": np.concatenate([_FRANKA_Q_MIN, _FRANKA_LIFT_CUBE_Q_MIN]),
}
"""Default lower generalized-position sampling bounds."""

JOINT_Q_MAX = {
    "Cartpole": np.array([1.0, np.pi]) if CARTPOLE_SMALL_RANGE else np.array([3.5, np.pi]),
    "Anymal-C": np.array(
        [
            10.0,
            1.0,
            10.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.72,
            -np.pi,
            -np.pi,
            0.49,
            -np.pi,
            -np.pi,
            0.72,
            -np.pi,
            -np.pi,
            0.49,
            -np.pi,
            -np.pi,
        ]
    ),
    "Franka": _FRANKA_Q_MAX,
    "Franka-Lift": np.concatenate([_FRANKA_Q_MAX, _FRANKA_LIFT_CUBE_Q_MAX]),
}
"""Default upper generalized-position sampling bounds."""

JOINT_QD_LIM = {
    "Cartpole": 1.0 if CARTPOLE_SMALL_RANGE else 10.0,
    "Anymal-C": np.array(
        [
            np.pi,
            np.pi,
            np.pi,
            0.25,
            0.25,
            0.25,
            np.pi * 2.0,
            np.pi * 2.0,
            np.pi * 2.0,
            np.pi * 2.0,
            np.pi * 2.0,
            np.pi * 2.0,
            np.pi * 2.0,
            np.pi * 2.0,
            np.pi * 2.0,
            np.pi * 2.0,
            np.pi * 2.0,
            np.pi * 2.0,
        ]
    ),
    "Franka": _FRANKA_QD_LIM,
    "Franka-Lift": np.concatenate([_FRANKA_QD_LIM, _FRANKA_LIFT_CUBE_QD_LIM]),
}
"""Default generalized-velocity sampling bounds."""

JOINT_F_LIM = {
    "Cartpole": np.array([200.0, 0.0]),
    "Anymal-C": 1.5
    * np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            50.0,
            40.0,
            8.0,
            50.0,
            40.0,
            8.0,
            50.0,
            40.0,
            8.0,
            50.0,
            40.0,
            8.0,
        ]
    ),
    "Franka": _FRANKA_F_LIM,
    "Franka-Lift": np.concatenate([_FRANKA_F_LIM, _FRANKA_LIFT_CUBE_F_LIM]),
}
"""Default generalized joint-force sampling bounds [N or N*m]."""

ANYMAL_C_KP_RANGE = (20.0, 80.0)
"""Default Anymal-C actuator stiffness randomization range."""

ANYMAL_C_KD_RANGE = (0.5, 4.0)
"""Default Anymal-C actuator damping randomization range."""

__all__ = [
    "ANYMAL_C_KD_RANGE",
    "ANYMAL_C_KP_RANGE",
    "DATASET_MODES",
    "JOINT_F_LIM",
    "JOINT_Q_MAX",
    "JOINT_Q_MIN",
    "JOINT_QD_LIM",
]
