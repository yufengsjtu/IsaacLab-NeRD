# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .abstract_contact import AbstractContact
from .active15_contact_encoder import Active15ContactEncoder
from .contact_set_encoder import (
    ContactSetEncoder,
    contact_token_validity_mask,
    mask_invalid_contact_tokens,
    transform_contact_tokens_to_body_frame,
)
from .contact_set_schema import (
    ACTIVE15_FEATURE_DIM,
    ACTIVE15_FEATURE_NAMES,
    ACTIVE15_TOKEN_DIM,
    CONTACT_REPRESENTATION_ACTIVE15,
    CONTACT_REPRESENTATION_ACTIVE15_SELF,
    CONTACT_REPRESENTATION_FLAT,
    CONTACT_REPRESENTATION_RAW15,
    CONTACT_REPRESENTATION_RAW15_SELF,
    CONTACT_REPRESENTATION_TOKENS,
    CONTACT_TOKEN_DIM,
    CONTACT_TOKEN_FEATURE_NAMES,
    DEFAULT_MAX_CONTACT_TOKENS,
    RAW15_FEATURE_DIM,
    RAW15_FEATURE_NAMES,
    RAW15_TOKEN_DIM,
)
from .contact_utils import collision_detection_fixed_ground, find_ground_shape_index
from .native15_self_contact_encoder import Active15SelfContactEncoder, Raw15SelfContactEncoder
from .newton_contact_adapter import NewtonContactAdapter
from .packing import ContactPackingPolicy, get_contact_order, resolve_contact_packing_policy
from .raw15_contact_encoder import Raw15ContactEncoder

__all__ = [
    "AbstractContact",
    "ACTIVE15_FEATURE_DIM",
    "ACTIVE15_FEATURE_NAMES",
    "ACTIVE15_TOKEN_DIM",
    "Active15ContactEncoder",
    "Active15SelfContactEncoder",
    "CONTACT_REPRESENTATION_ACTIVE15",
    "CONTACT_REPRESENTATION_ACTIVE15_SELF",
    "CONTACT_REPRESENTATION_FLAT",
    "CONTACT_REPRESENTATION_RAW15",
    "CONTACT_REPRESENTATION_RAW15_SELF",
    "CONTACT_REPRESENTATION_TOKENS",
    "CONTACT_TOKEN_DIM",
    "CONTACT_TOKEN_FEATURE_NAMES",
    "ContactPackingPolicy",
    "ContactSetEncoder",
    "DEFAULT_MAX_CONTACT_TOKENS",
    "NewtonContactAdapter",
    "RAW15_FEATURE_DIM",
    "RAW15_FEATURE_NAMES",
    "RAW15_TOKEN_DIM",
    "Raw15ContactEncoder",
    "Raw15SelfContactEncoder",
    "collision_detection_fixed_ground",
    "contact_token_validity_mask",
    "find_ground_shape_index",
    "get_contact_order",
    "mask_invalid_contact_tokens",
    "resolve_contact_packing_policy",
    "transform_contact_tokens_to_body_frame",
]
