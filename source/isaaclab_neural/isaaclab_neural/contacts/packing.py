from __future__ import annotations

from typing import Literal

import torch

ContactPackingPolicy = Literal[
    "stable_index",
    "force_priority",
    "penetration_priority",
    "random",
]


def get_contact_order(
    raw_contacts: dict[str, torch.Tensor],
    packing_policy: ContactPackingPolicy = "stable_index",
) -> torch.Tensor:
    """Return contact indices in the order they should be packed.

    Args:
        raw_contacts: Contact tensors read from Newton native contacts. Required
            keys are ``depth``, ``thickness0``, and ``thickness1``.
        packing_policy: Policy used to choose which contacts occupy fixed slots.

    Returns:
        A 1-D tensor of contact indices.
    """
    depth = raw_contacts["depth"]
    count = depth.shape[0]
    device = depth.device

    if packing_policy == "stable_index":
        return torch.arange(count, device=device)

    separation = depth - raw_contacts["thickness0"] - raw_contacts["thickness1"]

    if packing_policy == "penetration_priority":
        return torch.argsort(separation)

    if packing_policy == "random":
        return torch.randperm(count, device=device)

    if packing_policy == "force_priority":
        # Native contacts do not always carry forces before solver execution.
        # Fall back to deepest-first, which is deterministic and always available.
        return torch.argsort(separation)

    raise ValueError(f"Unsupported contact packing policy: {packing_policy}")
