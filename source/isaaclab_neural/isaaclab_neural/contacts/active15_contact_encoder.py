# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pack active Newton contacts into owner-body-frame Active15 tokens."""

from __future__ import annotations

import newton
import torch

from isaaclab_neural.contacts.contact_set_encoder import ContactSetEncoder, _as_torch_array, _segment_rank
from isaaclab_neural.contacts.contact_set_schema import (
    ACTIVE15_TOKEN_DIM,
    ACTIVE15_VALID_INDEX,
)
from isaaclab_neural.utils import torch_utils


class Active15ContactEncoder(ContactSetEncoder):
    """Encode one owner-routed token for each solver-active robot contact."""

    _solver_active_only = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        shape_margin = getattr(self.model, "shape_margin", None)
        if shape_margin is None:
            raise ValueError("Native15 contact encoding requires model.shape_margin.")
        self._shape_margin = _as_torch_array(shape_margin, self.device).reshape(-1).to(dtype=torch.float32)

    def encode(
        self,
        raw_contacts: dict[str, torch.Tensor],
        state: newton.State,
    ) -> torch.Tensor:
        """Return padded [num_envs, max_contact_tokens, 17] owner-frame native15 tokens."""
        count = raw_contacts["shape0"].shape[0]
        if count == 0:
            return self._empty_packed()

        shape0 = raw_contacts["shape0"]
        shape1 = raw_contacts["shape1"]
        body0 = self._shape_body_ids(shape0)
        body1 = self._shape_body_ids(shape1)
        primary0 = self._is_primary_body(body0)
        primary1 = self._is_primary_body(body1)
        owner_slot = self._body_slot(body0)
        world_id = self._contact_env_ids(body0, shape0)

        point0_world = raw_contacts["point0_world"]
        point1_world = raw_contacts["point1_world"]
        normal01_world = raw_contacts["normal"]
        owner_margin = raw_contacts["thickness0"]
        other_margin = raw_contacts["thickness1"]
        shape_margin0 = self._shape_margins(shape0)
        shape_margin1 = self._shape_margins(shape1)
        effective_radius0 = owner_margin - shape_margin0
        effective_radius1 = other_margin - shape_margin1
        signed_gap = torch.sum(normal01_world * (point1_world - point0_world), dim=-1)
        signed_gap = signed_gap - effective_radius0 - effective_radius1
        clearance = signed_gap - shape_margin0 - shape_margin1

        body_q = _as_torch_array(state.body_q, self.device)
        body_qd = _as_torch_array(state.body_qd, self.device)
        safe_owner = body0.clamp(min=0, max=body_q.shape[0] - 1)
        owner_pose = body_q.index_select(0, safe_owner)
        owner_position = owner_pose[:, :3]
        owner_rotation = owner_pose[:, 3:7]
        owner_point = torch_utils.transform_point_inverse(owner_position, owner_rotation, point0_world)
        other_point = torch_utils.transform_point_inverse(owner_position, owner_rotation, point1_world)
        owner_normal = torch_utils.quat_rotate_inverse(owner_rotation, -normal01_world)

        midpoint_world = 0.5 * (point0_world + point1_world)
        owner_com_world = self._body_com_world(body_q, body0)
        other_com_world = self._body_com_world(body_q, body1)
        owner_velocity = self._surface_velocity(body0 >= 0, body0, body_qd, midpoint_world, owner_com_world)
        other_velocity = self._surface_velocity(body1 >= 0, body1, body_qd, midpoint_world, other_com_world)
        relative_velocity = torch_utils.quat_rotate_inverse(
            owner_rotation,
            owner_velocity - other_velocity,
        )

        # The adapter canonicalizes a unique primary side to side 0. Both
        # native15 views exclude robot self-collisions; Active15 additionally
        # applies Newton's strict solver-active gate.
        valid = primary0 & ~primary1 & (owner_slot >= 0) & (world_id >= 0) & (world_id < self.num_envs)
        if self._solver_active_only:
            valid &= clearance < 0.0
        candidates = torch.cat(
            (
                valid.to(torch.float32).unsqueeze(-1),
                owner_slot.to(torch.float32).unsqueeze(-1),
                owner_point,
                other_point,
                owner_normal,
                signed_gap.unsqueeze(-1),
                relative_velocity,
                owner_margin.unsqueeze(-1),
                other_margin.unsqueeze(-1),
            ),
            dim=-1,
        )
        packed, overflow, body_ids = self._pack_active_rows(
            candidates,
            valid,
            world_id,
            body0,
            signed_gap,
        )
        if self._overflow is None or self._overflow.device != packed.device:
            self._overflow = torch.zeros(self.num_envs, dtype=torch.long, device=packed.device)
        self._overflow.copy_(overflow)
        self._body_ids = body_ids
        self._overflow_total += int(overflow.sum().item())
        return packed

    def _empty_packed(self) -> torch.Tensor:
        packed = torch.zeros(
            (self.num_envs, self.max_contact_tokens, ACTIVE15_TOKEN_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self._overflow = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._body_ids = torch.full(
            (self.num_envs, self.max_contact_tokens),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        return packed

    def _shape_margins(self, shape_ids: torch.Tensor) -> torch.Tensor:
        safe = shape_ids.clamp(min=0, max=self._shape_margin.shape[0] - 1)
        values = self._shape_margin.index_select(0, safe)
        return torch.where(shape_ids >= 0, values, torch.zeros_like(values))

    def _is_primary_body(self, body_ids: torch.Tensor) -> torch.Tensor:
        primary = torch.zeros_like(body_ids, dtype=torch.bool)
        valid = (body_ids >= 0) & (body_ids < self.primary_body_mask.shape[0])
        primary[valid] = self.primary_body_mask[body_ids[valid]]
        return primary

    def _pack_active_rows(
        self,
        candidates: torch.Tensor,
        valid: torch.Tensor,
        world_id: torch.Tensor,
        body_id: torch.Tensor,
        gap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack contacts round-robin by owner body, then by signed gap."""
        capacity = self.max_contact_tokens
        dump_world = self.num_envs
        invalid_body = int(self.model.body_count)
        body_key = torch.where(valid, body_id, torch.full_like(body_id, invalid_body))
        gap_key = torch.where(valid, gap, torch.full_like(gap, float("inf")))
        body_order = torch.argsort(gap_key, stable=True)
        body_order = body_order.index_select(
            0,
            torch.argsort(body_key.index_select(0, body_order), stable=True),
        )
        body_rank = torch.empty_like(body_id)
        body_rank.index_copy_(
            0,
            body_order,
            _segment_rank(body_key.index_select(0, body_order), gap.new_zeros(gap.shape)),
        )

        world_key = torch.where(valid, world_id, torch.full_like(world_id, dump_world))
        order = torch.argsort(gap_key, stable=True)
        order = order.index_select(0, torch.argsort(body_rank.index_select(0, order), stable=True))
        order = order.index_select(0, torch.argsort(world_key.index_select(0, order), stable=True))
        sorted_world = world_key.index_select(0, order)
        sorted_rank = _segment_rank(sorted_world, gap.new_zeros(gap.shape))
        keep_sorted = valid.index_select(0, order) & (sorted_world < self.num_envs) & (sorted_rank < capacity)
        destination_sorted = torch.where(
            keep_sorted,
            sorted_world * capacity + sorted_rank,
            torch.full_like(sorted_rank, self.num_envs * capacity),
        )

        flat = candidates.new_zeros((self.num_envs * capacity + 1, ACTIVE15_TOKEN_DIM))
        flat.index_copy_(0, destination_sorted, candidates.index_select(0, order))
        packed = flat[:-1].reshape(self.num_envs, capacity, ACTIVE15_TOKEN_DIM)
        packed[..., ACTIVE15_VALID_INDEX] = (packed[..., ACTIVE15_VALID_INDEX] > 0.5).to(torch.float32)

        flat_body_ids = torch.full(
            (self.num_envs * capacity + 1,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        flat_body_ids.index_copy_(0, destination_sorted, body_id.index_select(0, order))
        packed_body_ids = flat_body_ids[:-1].reshape(self.num_envs, capacity)
        overflow = torch.zeros(self.num_envs + 1, dtype=torch.long, device=self.device)
        dropped_sorted = valid.index_select(0, order) & ~keep_sorted
        if dropped_sorted.any():
            overflow.scatter_add_(
                0,
                sorted_world[dropped_sorted],
                torch.ones_like(sorted_world[dropped_sorted]),
            )
        return packed, overflow[: self.num_envs], packed_body_ids
