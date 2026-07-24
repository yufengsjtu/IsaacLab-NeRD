# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

import torch

ContactPackingPolicy = Literal[
    "stable_index",
    "force_priority",
    "penetration_priority",
    "random",
    "body_round_robin_pair_atomic",
]


def resolve_contact_packing_policy(
    contact_mode: Literal["fixed_ground", "newton_native"],
    packing_policy: ContactPackingPolicy | None,
) -> ContactPackingPolicy:
    """Resolve the default packing policy for one contact mode."""
    if packing_policy is not None:
        return packing_policy
    if contact_mode == "newton_native":
        return "penetration_priority"
    return "stable_index"


def get_contact_order(
    raw_contacts: dict[str, torch.Tensor],
    packing_policy: ContactPackingPolicy = "stable_index",
) -> torch.Tensor:
    """Return contact indices in the order they should be packed.

    Args:
        raw_contacts: Contact tensors read from Newton native contacts. Required
            key is ``surface_separation``.
        packing_policy: Policy used to choose which contacts occupy fixed slots.

    Returns:
        A 1-D tensor of contact indices.
    """
    separation = raw_contacts["surface_separation"]
    count = separation.shape[0]
    device = separation.device

    if packing_policy == "stable_index":
        return torch.arange(count, device=device)

    if packing_policy == "penetration_priority":
        return _penetration_priority_order(raw_contacts)

    if packing_policy == "random":
        return torch.randperm(count, device=device)

    if packing_policy == "force_priority":
        # Native contacts do not always carry forces before solver execution.
        # Fall back to deepest-first, which is deterministic and always available.
        return _penetration_priority_order(raw_contacts)

    if packing_policy == "body_round_robin_pair_atomic":
        raise ValueError(
            "Packing policy 'body_round_robin_pair_atomic' is only used by the "
            "contact_tokens representation inside ContactSetEncoder. For flat "
            "newton_native packing choose penetration_priority, stable_index, "
            "random, or force_priority."
        )

    raise ValueError(f"Unsupported contact packing policy: {packing_policy}")


def _penetration_priority_order(raw_contacts: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sort deepest-first with deterministic geometry-based tie breaks."""
    separation = raw_contacts["surface_separation"]
    order = torch.arange(separation.shape[0], device=separation.device)
    tie_break_fields = (
        raw_contacts["point1_world"],
        raw_contacts["point0_world"],
    )

    # Stable sorts are applied from least- to most-significant key.
    for points in tie_break_fields:
        quantized_points = torch.round(points * 1.0e6).to(torch.int64)
        for axis in reversed(range(quantized_points.shape[-1])):
            indices = torch.argsort(quantized_points[order, axis], stable=True)
            order = order[indices]
    for name in ("shape1", "shape0"):
        indices = torch.argsort(raw_contacts[name][order], stable=True)
        order = order[indices]

    quantized_separation = torch.round(separation * 1.0e6).to(torch.int64)
    indices = torch.argsort(quantized_separation[order], stable=True)
    return order[indices]
