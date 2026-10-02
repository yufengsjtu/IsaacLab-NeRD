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
CONTACT_FILTER_SOLVER_ACTIVE_MOVABLE = "solver_active_movable"

CONTACT_REPRESENTATION_FLAT = "flat"
CONTACT_REPRESENTATION_TOKENS = "contact_tokens"

CONTACT_FLAT_VALUE_FIELDS: tuple[str, ...] = (
    "contact_normals",
    "contact_depths",
    "contact_thicknesses_0",
    "contact_thicknesses_1",
    "contact_points_0",
    "contact_points_1",
)

NATIVE_CONTACT_FRAME_COUNTS = "frame_counts"
NATIVE_CONTACT_DATASET_SCHEMA_VERSION = 3
NATIVE_CONTACT_STORAGE_SCHEMA = "native_pairs_csr_v1"
NATIVE_CONTACT_FRAME = "world_v1"

NATIVE_CONTACT_INTEGER_FIELDS: tuple[str, ...] = (
    "shape_ids_0",
    "shape_ids_1",
    "body_ids_0",
    "body_ids_1",
    "body_slots_0",
    "body_slots_1",
    "world_ids",
)

NATIVE_CONTACT_VECTOR_FIELDS: tuple[str, ...] = (
    "points_0_local",
    "points_1_local",
    "points_0_w",
    "points_1_w",
    "surface_points_0_w",
    "surface_points_1_w",
    "normal_w",
    "offsets_0",
    "offsets_1",
    "lever_arms_0_w",
    "lever_arms_1_w",
    "relative_velocities_0_w",
    "relative_velocities_1_w",
)

NATIVE_CONTACT_SCALAR_FIELDS: tuple[str, ...] = (
    "separation",
    "thickness_0",
    "thickness_1",
)

NATIVE_CONTACT_FLOAT_FIELDS: tuple[str, ...] = NATIVE_CONTACT_VECTOR_FIELDS + NATIVE_CONTACT_SCALAR_FIELDS
NATIVE_CONTACT_VALUE_FIELDS: tuple[str, ...] = NATIVE_CONTACT_INTEGER_FIELDS + NATIVE_CONTACT_FLOAT_FIELDS

NATIVE_CONTACT_PRECISION_ATTR = "contact_value_precision"
"""Native-pair group attribute naming the stored value precision (absent means full precision)."""

NATIVE_CONTACT_PRECISION_FULL = "full"
NATIVE_CONTACT_PRECISION_COMPACT = "compact_v1"

NATIVE_CONTACT_COMPACT_DTYPES: dict[str, str] = {
    "shape_ids_0": "int32",
    "shape_ids_1": "int32",
    "body_ids_0": "int32",
    "body_ids_1": "int32",
    "body_slots_0": "int16",
    "body_slots_1": "int16",
    "world_ids": "int16",
    # Positions, separations and relative velocities drive filtering, packing and contact dynamics: keep FP32.
    "points_0_w": "float32",
    "points_1_w": "float32",
    "surface_points_0_w": "float32",
    "surface_points_1_w": "float32",
    "separation": "float32",
    "relative_velocities_0_w": "float32",
    "relative_velocities_1_w": "float32",
    # Unit normals, body-local offsets, lever arms [m] and margins [m] are bounded and tolerate FP16.
    "points_0_local": "float16",
    "points_1_local": "float16",
    "normal_w": "float16",
    "offsets_0": "float16",
    "offsets_1": "float16",
    "lever_arms_0_w": "float16",
    "lever_arms_1_w": "float16",
    "thickness_0": "float16",
    "thickness_1": "float16",
}
"""Stored dtypes of ``compact_v1`` native pairs; readers widen them to int64/float32."""
