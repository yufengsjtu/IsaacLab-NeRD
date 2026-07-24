# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .abstract_contact import AbstractContact
from .contact_set_encoder import (
    ContactSetEncoder,
    contact_token_validity_mask,
    mask_invalid_contact_tokens,
    transform_contact_tokens_to_body_frame,
)
from .contact_set_schema import (
    CONTACT_REPRESENTATION_FLAT,
    CONTACT_REPRESENTATION_TOKENS,
    CONTACT_TOKEN_DIM,
    CONTACT_TOKEN_FEATURE_NAMES,
    DEFAULT_MAX_CONTACT_TOKENS,
)
from .contact_utils import collision_detection_fixed_ground, find_ground_shape_index
from .newton_contact_adapter import NewtonContactAdapter
from .packing import ContactPackingPolicy, get_contact_order, resolve_contact_packing_policy

__all__ = [
    "AbstractContact",
    "CONTACT_REPRESENTATION_FLAT",
    "CONTACT_REPRESENTATION_TOKENS",
    "CONTACT_TOKEN_DIM",
    "CONTACT_TOKEN_FEATURE_NAMES",
    "ContactPackingPolicy",
    "ContactSetEncoder",
    "DEFAULT_MAX_CONTACT_TOKENS",
    "NewtonContactAdapter",
    "collision_detection_fixed_ground",
    "contact_token_validity_mask",
    "find_ground_shape_index",
    "get_contact_order",
    "mask_invalid_contact_tokens",
    "resolve_contact_packing_policy",
    "transform_contact_tokens_to_body_frame",
]
