# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Schema constants for PhysicsNeMo-compatible contact token sets."""

from __future__ import annotations

CONTACT_TOKEN_DIM = 17

CONTACT_TOKEN_FEATURE_NAMES: tuple[str, ...] = (
    "valid",
    "body_slot",
    "other_body_slot",
    "other_is_dynamic",
    "point_x",
    "point_y",
    "point_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "lever_arm_x",
    "lever_arm_y",
    "lever_arm_z",
    "gap",
    "relative_velocity_x",
    "relative_velocity_y",
    "relative_velocity_z",
)

CONTACT_TOKEN_CATEGORICAL_CHANNELS: tuple[int, ...] = (0, 1, 2, 3)

CONTACT_TOKEN_GEOMETRY_SLICE = slice(4, CONTACT_TOKEN_DIM)

CONTACT_TOKEN_VALID_INDEX = 0
CONTACT_TOKEN_BODY_SLOT_INDEX = 1
CONTACT_TOKEN_OTHER_BODY_SLOT_INDEX = 2
CONTACT_TOKEN_OTHER_DYNAMIC_INDEX = 3
CONTACT_TOKEN_GAP_INDEX = 13

DEFAULT_MAX_CONTACT_TOKENS = 128

CONTACT_REPRESENTATION_FLAT = "flat"
CONTACT_REPRESENTATION_TOKENS = "contact_tokens"
