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

DEFAULT_MAX_CONTACT_TOKENS = 64
CONTACT_FILTER_NONE = "none"
CONTACT_FILTER_SOLVER_ACTIVE = "solver_active"
CONTACT_TOKEN_SOLVER_ACTIVE_FIELD = "contact_token_solver_active"
CONTACT_TOKEN_SOLVER_ACTIVE_SCHEMA = "mujoco_solver_included_v1"


CONTACT_REPRESENTATION_FLAT = "flat"
CONTACT_REPRESENTATION_TOKENS = "contact_tokens"
CONTACT_REPRESENTATION_ACTIVE15 = "active15_tokens"
CONTACT_REPRESENTATION_RAW15 = "raw15_tokens"

ACTIVE15_TOKEN_DIM = 17
ACTIVE15_FEATURE_DIM = 15
ACTIVE15_CATEGORICAL_CHANNELS: tuple[int, ...] = (0, 1)
ACTIVE15_FEATURE_SLICE = slice(2, ACTIVE15_TOKEN_DIM)
ACTIVE15_VALID_INDEX = 0
ACTIVE15_BODY_SLOT_INDEX = 1
ACTIVE15_GAP_INDEX = 11
ACTIVE15_OWNER_POINT_SLICE = slice(2, 5)
ACTIVE15_OTHER_POINT_SLICE = slice(5, 8)
ACTIVE15_OWNER_NORMAL_SLICE = slice(8, 11)
ACTIVE15_OWNER_MARGIN_INDEX = 15
ACTIVE15_OTHER_MARGIN_INDEX = 16

ACTIVE15_FEATURE_NAMES: tuple[str, ...] = (
    "owner_point_x",
    "owner_point_y",
    "owner_point_z",
    "other_point_x",
    "other_point_y",
    "other_point_z",
    "owner_directed_normal_x",
    "owner_directed_normal_y",
    "owner_directed_normal_z",
    "signed_solver_gap",
    "relative_velocity_x",
    "relative_velocity_y",
    "relative_velocity_z",
    "owner_margin",
    "other_margin",
)


# Raw15 and Active15 have the same owner-frame fields. Their only semantic
# difference is whether the upstream extractor applies the solver-active gate.
RAW15_TOKEN_DIM = ACTIVE15_TOKEN_DIM
RAW15_FEATURE_DIM = ACTIVE15_FEATURE_DIM
RAW15_CATEGORICAL_CHANNELS = ACTIVE15_CATEGORICAL_CHANNELS
RAW15_FEATURE_SLICE = ACTIVE15_FEATURE_SLICE
RAW15_VALID_INDEX = ACTIVE15_VALID_INDEX
RAW15_BODY_SLOT_INDEX = ACTIVE15_BODY_SLOT_INDEX
RAW15_GAP_INDEX = ACTIVE15_GAP_INDEX
RAW15_FEATURE_NAMES = ACTIVE15_FEATURE_NAMES


def is_native15_contact_representation(representation: str) -> bool:
    """Return whether the representation uses the owner-frame native15 schema."""
    return representation in {CONTACT_REPRESENTATION_RAW15, CONTACT_REPRESENTATION_ACTIVE15}


def is_contact_token_representation(representation: str) -> bool:
    """Return whether the representation uses padded contact-token transport."""
    return representation == CONTACT_REPRESENTATION_TOKENS or is_native15_contact_representation(representation)
