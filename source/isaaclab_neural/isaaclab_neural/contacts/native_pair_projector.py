# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project variable native contact pairs into fixed-capacity neural inputs."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_FILTER_NONE,
    CONTACT_FILTER_SOLVER_ACTIVE,
    CONTACT_FILTER_SOLVER_ACTIVE_MOVABLE,
    CONTACT_REPRESENTATION_FLAT,
    CONTACT_REPRESENTATION_TOKENS,
    CONTACT_TOKEN_DIM,
)
from isaaclab_neural.contacts.packing import ContactPackingPolicy


def native_pair_solver_active_mask(native_pairs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Select negative recorded surface separation [m], not a measured-force threshold."""
    if "separation" not in native_pairs:
        raise ValueError("Solver-active classification requires native field 'separation'.")
    return native_pairs["separation"] < 0.0


def gather_or_fill(table: torch.Tensor, indices: torch.Tensor, fill: int | bool = -1) -> torch.Tensor:
    """Return ``table[indices]`` where ``0 <= indices < len(table)`` and ``fill`` elsewhere.

    Unlike boolean-mask gathers, this issues no host synchronization and keeps the input shape.
    """
    if table.shape[0] == 0:
        return torch.full(indices.shape, fill, dtype=table.dtype, device=indices.device)
    valid = (indices >= 0) & (indices < table.shape[0])
    gathered = table.index_select(0, indices.reshape(-1).long().clamp(0, table.shape[0] - 1)).reshape(indices.shape)
    return torch.where(valid, gathered, torch.full_like(gathered, fill))


_FIXED_SLOT_TENSORS: dict[tuple[tuple[int, ...], torch.device], torch.Tensor] = {}


def native_pair_filter_mask(
    native_pairs: Mapping[str, torch.Tensor], contact_filter: str, fixed_body_slots: tuple[int, ...] | None = None
) -> torch.Tensor:
    """Select input pairs without modifying solver contacts or using future force labels."""
    if contact_filter == CONTACT_FILTER_NONE:
        return torch.ones_like(native_pairs["separation"], dtype=torch.bool)
    active = native_pair_solver_active_mask(native_pairs)
    if contact_filter == CONTACT_FILTER_SOLVER_ACTIVE:
        return active
    if contact_filter != CONTACT_FILTER_SOLVER_ACTIVE_MOVABLE:
        raise ValueError(f"Unsupported contact filter: {contact_filter!r}.")
    if fixed_body_slots is None or any(type(slot) is not int or slot < 0 for slot in fixed_body_slots):
        raise ValueError("Movable contact filtering requires verified fixed body slots (mobility metadata).")
    key = (tuple(fixed_body_slots), active.device)
    fixed = _FIXED_SLOT_TENSORS.get(key)
    if fixed is None:
        # Cached: re-uploading the slot list every frame cost a host-to-device copy per step.
        fixed = _FIXED_SLOT_TENSORS[key] = torch.tensor(fixed_body_slots, device=active.device, dtype=torch.long)
    immovable = [
        (native_pairs[f"body_ids_{side}"] < 0) | torch.isin(native_pairs[f"body_slots_{side}"], fixed)
        for side in (0, 1)
    ]
    return active & ~(immovable[0] & immovable[1])


def _stable_segment_rank(keys: torch.Tensor) -> torch.Tensor:
    """Return the encounter-order rank within each key segment."""
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys.index_select(0, order)
    if sorted_keys.numel() == 0:
        return sorted_keys
    positions = torch.arange(sorted_keys.numel(), device=keys.device)
    changes = torch.ones_like(sorted_keys, dtype=torch.bool)
    changes[1:] = sorted_keys[1:] != sorted_keys[:-1]
    starts = torch.where(changes, positions, 0)
    starts = torch.cummax(starts, dim=0).values
    rank = torch.empty_like(positions)
    rank.index_copy_(0, order, positions - starts)
    return rank


def _segment_exclusive_sum(segment_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Return exclusive prefix sums grouped by segment id."""
    if segment_ids.numel() == 0:
        return values.new_zeros((0,))
    order = torch.argsort(segment_ids, stable=True)
    sorted_ids = segment_ids.index_select(0, order)
    sorted_values = values.index_select(0, order)
    changes = torch.ones_like(sorted_ids, dtype=torch.bool)
    changes[1:] = sorted_ids[1:] != sorted_ids[:-1]
    positions = torch.arange(sorted_ids.numel(), device=segment_ids.device)
    starts = positions[changes]
    inclusive = torch.cumsum(sorted_values, dim=0)
    bases = torch.zeros_like(starts, dtype=inclusive.dtype)
    if starts.numel() > 1:
        bases[1:] = inclusive.index_select(0, starts[1:] - 1)
    segment_index = torch.cumsum(changes.to(torch.long), dim=0) - 1
    exclusive = inclusive - sorted_values - bases.index_select(0, segment_index)
    result = torch.empty_like(exclusive)
    result.index_copy_(0, order, exclusive)
    return result


def _filter_native_pairs(
    native_pairs: Mapping[str, torch.Tensor],
    frame_ids: torch.Tensor,
    contact_filter: str,
    fixed_body_slots: tuple[int, ...] | None,
) -> tuple[Mapping[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Select native pairs admitted by the requested contact filter."""
    source_indices = torch.arange(frame_ids.shape[0], device=frame_ids.device)
    if contact_filter == CONTACT_FILTER_NONE:
        return native_pairs, frame_ids, source_indices
    active = native_pair_filter_mask(native_pairs, contact_filter, fixed_body_slots)
    indices = torch.nonzero(active, as_tuple=False).reshape(-1)
    return (
        {name: value.index_select(0, indices) for name, value in native_pairs.items()},
        frame_ids.index_select(0, indices),
        indices,
    )


def _flat_order(native_pairs: Mapping[str, torch.Tensor], packing_policy: ContactPackingPolicy) -> torch.Tensor:
    """Return the legacy flat contact order."""
    separation = native_pairs["separation"]
    count = separation.shape[0]
    order = torch.arange(count, device=separation.device)
    if packing_policy == "stable_index":
        return order
    if packing_policy == "random":
        return torch.randperm(count, device=separation.device)
    if packing_policy not in {"penetration_priority", "force_priority"}:
        raise ValueError(f"Unsupported flat contact packing policy: {packing_policy!r}.")

    # Stable sorts are applied from least- to most-significant key. This is the
    # exact ordering used by the former runtime flat packer.
    for points in (native_pairs["points_1_w"], native_pairs["points_0_w"]):
        quantized = torch.round(points * 1.0e6).to(torch.int64)
        for axis in reversed(range(quantized.shape[-1])):
            order = order.index_select(0, torch.argsort(quantized[order, axis], stable=True))
    for name in ("shape_ids_1", "shape_ids_0"):
        order = order.index_select(0, torch.argsort(native_pairs[name][order], stable=True))
    quantized_separation = torch.round(separation * 1.0e6).to(torch.int64)
    return order.index_select(0, torch.argsort(quantized_separation[order], stable=True))


def _project_flat(
    native_pairs: Mapping[str, torch.Tensor],
    frame_ids: torch.Tensor,
    num_frames: int,
    capacity: int,
    packing_policy: ContactPackingPolicy,
    return_pair_indices: bool,
) -> dict[str, torch.Tensor]:
    """Project native pairs to legacy flat fields."""
    device = frame_ids.device
    result = {
        "contact_masks": torch.zeros((num_frames, capacity), dtype=torch.bool, device=device),
        "contact_normals": torch.zeros((num_frames, capacity, 3), dtype=torch.float32, device=device),
        "contact_depths": torch.zeros((num_frames, capacity), dtype=torch.float32, device=device),
        "contact_thicknesses_0": torch.zeros((num_frames, capacity), dtype=torch.float32, device=device),
        "contact_thicknesses_1": torch.zeros((num_frames, capacity), dtype=torch.float32, device=device),
        "contact_points_0": torch.zeros((num_frames, capacity, 3), dtype=torch.float32, device=device),
        "contact_points_1": torch.zeros((num_frames, capacity, 3), dtype=torch.float32, device=device),
    }
    if frame_ids.numel() == 0:
        result["contact_normals"] = result["contact_normals"].reshape(num_frames, capacity * 3)
        result["contact_points_0"] = result["contact_points_0"].reshape(num_frames, capacity * 3)
        result["contact_points_1"] = result["contact_points_1"].reshape(num_frames, capacity * 3)
        if return_pair_indices:
            result["_pair_indices"] = torch.full((num_frames, capacity), -1, dtype=torch.long, device=device)
        return result

    order = _flat_order(native_pairs, packing_policy)
    ordered_frames = frame_ids.index_select(0, order)
    valid = (ordered_frames >= 0) & (ordered_frames < num_frames)
    valid_order = order[valid]
    valid_frames = ordered_frames[valid]
    rank = _stable_segment_rank(valid_frames)
    keep = rank < capacity
    source = valid_order[keep]
    destinations = valid_frames[keep]
    slots = rank[keep]

    result["contact_masks"][destinations, slots] = True
    result["contact_normals"][destinations, slots] = native_pairs["normal_w"][source]
    result["contact_depths"][destinations, slots] = native_pairs["separation"][source]
    result["contact_thicknesses_0"][destinations, slots] = native_pairs["thickness_0"][source]
    result["contact_thicknesses_1"][destinations, slots] = native_pairs["thickness_1"][source]
    result["contact_points_0"][destinations, slots] = native_pairs["points_0_w"][source]
    result["contact_points_1"][destinations, slots] = native_pairs["points_1_w"][source]
    if return_pair_indices:
        pair_indices = torch.full((num_frames, capacity), -1, dtype=torch.long, device=device)
        pair_indices[destinations, slots] = source
        result["_pair_indices"] = pair_indices
    result["contact_normals"] = result["contact_normals"].reshape(num_frames, capacity * 3)
    result["contact_points_0"] = result["contact_points_0"].reshape(num_frames, capacity * 3)
    result["contact_points_1"] = result["contact_points_1"].reshape(num_frames, capacity * 3)
    return result


def _project_tokens(
    native_pairs: Mapping[str, torch.Tensor],
    frame_ids: torch.Tensor,
    num_frames: int,
    capacity: int,
    packing_policy: ContactPackingPolicy,
) -> dict[str, torch.Tensor]:
    """Project native pairs to directed, pair-atomic contact tokens."""
    if packing_policy != "body_round_robin_pair_atomic":
        raise ValueError("Contact-token projection requires packing_policy='body_round_robin_pair_atomic'.")
    device = frame_ids.device
    tokens = torch.zeros((num_frames, capacity, CONTACT_TOKEN_DIM), dtype=torch.float32, device=device)
    overflow = torch.zeros(num_frames, dtype=torch.long, device=device)
    packed_body_ids = torch.full((num_frames, capacity), -1, dtype=torch.long, device=device)
    packed_world_ids = torch.full((num_frames, capacity), -1, dtype=torch.long, device=device)
    result = {
        "contact_tokens": tokens,
        "contact_token_overflow": overflow,
        "contact_token_body_ids": packed_body_ids,
        "contact_token_world_ids": packed_world_ids,
    }
    pair_count = frame_ids.numel()
    if pair_count == 0:
        return result

    valid0 = native_pairs["body_slots_0"] >= 0
    valid1 = native_pairs["body_slots_1"] >= 0
    directed_valid = torch.cat((valid0, valid1))
    directed_frames = frame_ids.repeat(2)
    sides = valid0.long() + valid1.long()
    pair_valid = sides > 0
    directed_body_ids = torch.cat((native_pairs["body_ids_0"], native_pairs["body_ids_1"]))
    directed_gaps = native_pairs["separation"].repeat(2)
    invalid_body = torch.iinfo(torch.long).max
    frame_key = torch.where(directed_valid, directed_frames, num_frames)
    body_key = torch.where(directed_valid, directed_body_ids, invalid_body)
    gap_key = torch.where(directed_valid, directed_gaps, float("inf"))
    directed_order = torch.argsort(gap_key, stable=True)
    directed_order = directed_order.index_select(
        0, torch.argsort(body_key.index_select(0, directed_order), stable=True)
    )
    directed_order = directed_order.index_select(
        0, torch.argsort(frame_key.index_select(0, directed_order), stable=True)
    )
    sorted_frame_key = frame_key.index_select(0, directed_order)
    sorted_body_key = body_key.index_select(0, directed_order)
    positions = torch.arange(directed_order.numel(), device=device)
    changes = torch.ones_like(sorted_frame_key, dtype=torch.bool)
    changes[1:] = (sorted_frame_key[1:] != sorted_frame_key[:-1]) | (sorted_body_key[1:] != sorted_body_key[:-1])
    starts = torch.where(changes, positions, 0)
    starts = torch.cummax(starts, dim=0).values
    directed_rank = torch.empty_like(directed_body_ids)
    directed_rank.index_copy_(0, directed_order, positions - starts)
    directed_rank = torch.where(directed_valid, directed_rank, 2 * pair_count)
    pair_rank = torch.minimum(directed_rank[:pair_count], directed_rank[pair_count:])

    safe_frames = torch.where(
        (frame_ids >= 0) & (frame_ids < num_frames),
        frame_ids,
        torch.full_like(frame_ids, num_frames),
    )
    pair_gap = torch.where(pair_valid, native_pairs["separation"], float("inf"))
    pair_order = torch.argsort(pair_gap, stable=True)
    pair_order = pair_order.index_select(0, torch.argsort(pair_rank.index_select(0, pair_order), stable=True))
    pair_order = pair_order.index_select(0, torch.argsort(safe_frames.index_select(0, pair_order), stable=True))
    sorted_frames = safe_frames.index_select(0, pair_order)
    sorted_sides = sides.index_select(0, pair_order)
    sorted_offsets = _segment_exclusive_sum(sorted_frames, sorted_sides)
    keep_sorted = (sorted_frames < num_frames) & (sorted_offsets + sorted_sides <= capacity)
    offsets = torch.empty_like(sorted_offsets)
    offsets.index_copy_(0, pair_order, sorted_offsets)
    keep_pair = torch.empty_like(keep_sorted)
    keep_pair.index_copy_(0, pair_order, keep_sorted)

    directed_keep = torch.cat((keep_pair & valid0, keep_pair & valid1))
    within_pair = torch.cat((torch.zeros_like(valid0, dtype=torch.long), valid0.long()))
    directed_offsets = offsets.repeat(2) + within_pair

    body_slots = torch.cat((native_pairs["body_slots_0"], native_pairs["body_slots_1"]))
    other_slots = torch.cat((native_pairs["body_slots_1"], native_pairs["body_slots_0"]))
    other_body_ids = torch.cat((native_pairs["body_ids_1"], native_pairs["body_ids_0"]))
    other_dynamic = other_body_ids >= 0
    other_slots = torch.where(other_dynamic, other_slots, -1)
    points = torch.cat((native_pairs["surface_points_0_w"], native_pairs["surface_points_1_w"]))
    normals = torch.cat((native_pairs["normal_w"], -native_pairs["normal_w"]))
    levers = torch.cat((native_pairs["lever_arms_0_w"], native_pairs["lever_arms_1_w"]))
    velocities = torch.cat((native_pairs["relative_velocities_0_w"], native_pairs["relative_velocities_1_w"]))
    candidates = torch.cat(
        (
            directed_valid.to(torch.float32).unsqueeze(-1),
            body_slots.to(torch.float32).unsqueeze(-1),
            other_slots.to(torch.float32).unsqueeze(-1),
            other_dynamic.to(torch.float32).unsqueeze(-1),
            points,
            normals,
            levers,
            directed_gaps.unsqueeze(-1),
            velocities,
        ),
        dim=-1,
    )
    kept = torch.nonzero(directed_keep, as_tuple=False).reshape(-1)
    kept_frames = directed_frames[kept]
    kept_slots = directed_offsets[kept]
    tokens[kept_frames, kept_slots] = candidates[kept]
    packed_body_ids[kept_frames, kept_slots] = directed_body_ids[kept]
    packed_world_ids[kept_frames, kept_slots] = native_pairs["world_ids"].repeat(2)[kept]

    dropped = directed_valid & ~directed_keep
    valid_directed_frames = (directed_frames >= 0) & (directed_frames < num_frames)
    # Masked adds of zero instead of boolean selection, so no host synchronization.
    overflow.scatter_add_(
        0,
        directed_frames.clamp(0, max(num_frames - 1, 0)),
        (dropped & valid_directed_frames).to(torch.long),
    )
    return result


def project_native_pairs(
    native_pairs: Mapping[str, torch.Tensor],
    *,
    num_frames: int,
    representation: str,
    packing_policy: ContactPackingPolicy,
    frame_ids: torch.Tensor | None = None,
    flat_capacity: int | None = None,
    max_contact_tokens: int | None = None,
    return_pair_indices: bool = False,
    contact_filter: str = CONTACT_FILTER_NONE,
    fixed_body_slots: tuple[int, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Project complete native pairs into fixed-capacity model contact fields.

    Args:
        native_pairs: Canonical native-pair tensors. ``world_ids`` supplies the
            per-pair frame id unless :paramref:`frame_ids` is provided.
        num_frames: Number of output frames.
        representation: ``"flat"`` or ``"contact_tokens"``.
        packing_policy: Stable contact selection policy.
        frame_ids: Optional output frame id for each pair. This permits batched
            dataset projection while retaining native ``world_ids`` as token
            identity metadata.
        flat_capacity: Contacts per frame for flat projection.
        max_contact_tokens: Directed tokens per frame for token projection.
        return_pair_indices: Include private source-pair indices for flat
            runtime metadata maintenance.
        contact_filter: Native-pair selection applied before capacity packing.
            ``"solver_active"`` keeps pairs with negative recorded surface
            separation [m]. ``"solver_active_movable"`` additionally removes pairs
            whose two sides are immovable. It does not filter the native simulation.
        fixed_body_slots: Verified immovable primary-body slots, required for movable filtering.
            Untracked bodies with nonnegative body IDs are conservatively retained.

    Returns:
        Original fixed-capacity flat fields, or token, overflow, body-id, and
        world-id fields.
    """
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative.")
    if "world_ids" not in native_pairs:
        raise ValueError("native_pairs must contain world_ids.")
    pair_world_ids = native_pairs["world_ids"]
    if frame_ids is None:
        frame_ids = pair_world_ids
    frame_ids = frame_ids.to(device=pair_world_ids.device, dtype=torch.long)
    if frame_ids.ndim != 1 or frame_ids.shape[0] != pair_world_ids.shape[0]:
        raise ValueError("frame_ids must contain one id per native pair.")
    native_pairs = dict(native_pairs)
    if "separation" not in native_pairs:
        raise ValueError("Native-pair projection requires field 'separation'.")
    native_pairs, frame_ids, source_pair_indices = _filter_native_pairs(
        native_pairs, frame_ids, contact_filter, fixed_body_slots
    )

    if representation == CONTACT_REPRESENTATION_FLAT:
        if flat_capacity is None or flat_capacity <= 0:
            raise ValueError("flat_capacity must be positive for flat projection.")
        result = _project_flat(
            native_pairs,
            frame_ids,
            num_frames,
            int(flat_capacity),
            packing_policy,
            return_pair_indices,
        )
        if return_pair_indices:
            pair_indices = result["_pair_indices"]
            selected = pair_indices >= 0
            pair_indices[selected] = source_pair_indices[pair_indices[selected]]
        return result
    if representation == CONTACT_REPRESENTATION_TOKENS:
        if max_contact_tokens is None or max_contact_tokens <= 0:
            raise ValueError("max_contact_tokens must be positive for token projection.")
        return _project_tokens(native_pairs, frame_ids, num_frames, int(max_contact_tokens), packing_policy)
    raise ValueError(f"Unsupported contact representation: {representation!r}.")
