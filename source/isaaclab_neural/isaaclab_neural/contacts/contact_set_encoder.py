# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pack Newton native contacts into fixed-capacity directed contact token sets."""

from __future__ import annotations

from dataclasses import dataclass

import newton
import torch
import warp as wp

from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_REPRESENTATION_TOKENS,
    CONTACT_TOKEN_BODY_SLOT_INDEX,
    CONTACT_TOKEN_DIM,
    CONTACT_TOKEN_GEOMETRY_SLICE,
    CONTACT_TOKEN_VALID_INDEX,
)
from isaaclab_neural.contacts.native_pair_projector import gather_or_fill, project_native_pairs
from isaaclab_neural.utils import torch_utils


def _as_torch_array(array: torch.Tensor | wp.array, device: torch.device | str) -> torch.Tensor:
    """Convert a Warp or Torch body-state array to a Torch tensor on ``device``."""
    if isinstance(array, torch.Tensor):
        return array.to(device=device)
    return wp.to_torch(array).to(device=device)


@dataclass
class _DirectedContactRows:
    """One directed contact candidate per row before capacity packing."""

    valid: torch.Tensor
    world_id: torch.Tensor
    body_id: torch.Tensor
    other_body_id: torch.Tensor
    other_dynamic: torch.Tensor
    point_world: torch.Tensor
    normal_world: torch.Tensor
    lever_world: torch.Tensor
    gap: torch.Tensor
    velocity_world: torch.Tensor


def _segment_rank(keys: torch.Tensor, ramp: torch.Tensor) -> torch.Tensor:
    """Return per-row rank within segments defined by ``keys``."""
    del ramp
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys.index_select(0, order)
    if sorted_keys.numel() == 0:
        return sorted_keys
    changes = torch.ones_like(sorted_keys, dtype=torch.bool)
    changes[1:] = sorted_keys[1:] != sorted_keys[:-1]
    positions = torch.arange(sorted_keys.numel(), device=keys.device)
    segment_starts = torch.where(changes, positions, 0)
    segment_starts = torch.cummax(segment_starts, dim=0).values
    rank_sorted = positions - segment_starts
    rank = torch.empty_like(rank_sorted)
    rank.index_copy_(0, order, rank_sorted)
    return rank


def _segment_exclusive_sum(segment_ids: torch.Tensor, values: torch.Tensor, ramp: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum of ``values`` within each segment id."""
    del ramp
    if segment_ids.numel() == 0:
        return values.new_zeros((0,))
    order = torch.argsort(segment_ids, stable=True)
    sorted_ids = segment_ids.index_select(0, order)
    sorted_values = values.index_select(0, order)
    changes = torch.ones_like(sorted_ids, dtype=torch.bool)
    changes[1:] = sorted_ids[1:] != sorted_ids[:-1]
    positions = torch.arange(sorted_ids.numel(), device=segment_ids.device)
    segment_starts = positions[changes]
    inclusive = torch.cumsum(sorted_values, dim=0)
    segment_bases = torch.zeros_like(segment_starts, dtype=inclusive.dtype)
    if segment_starts.numel() > 1:
        segment_bases[1:] = inclusive.index_select(0, segment_starts[1:] - 1)
    segment_index = torch.cumsum(changes.to(torch.long), dim=0) - 1
    exclusive = inclusive - sorted_values - segment_bases.index_select(0, segment_index)
    result = torch.empty_like(exclusive)
    result.index_copy_(0, order, exclusive)
    return result


class ContactSetEncoder:
    """Encode Newton contacts as padded directed token sets."""

    def __init__(
        self,
        *,
        model: newton.Model,
        primary_body_mask: torch.Tensor,
        shape_body_torch: torch.Tensor,
        bodies_per_env: int,
        num_envs: int,
        max_contact_tokens: int,
        device: torch.device | str,
        primary_body_ids_by_world: torch.Tensor | None = None,
    ) -> None:
        if max_contact_tokens <= 0:
            raise ValueError("max_contact_tokens must be a positive integer.")
        self.model = model
        self.primary_body_mask = primary_body_mask
        self.shape_body_torch = shape_body_torch
        self.bodies_per_env = bodies_per_env
        self.num_envs = num_envs
        self.max_contact_tokens = int(max_contact_tokens)
        self.device = torch.device(device)
        self.primary_body_ids_by_world = primary_body_ids_by_world
        # Cache as int64: Newton exposes body_world as int32, but contact indexing
        # buffers are long and PyTorch index-put requires matching dtypes.
        if model.body_world is None:
            self.body_world = None
        else:
            self.body_world = torch.as_tensor(model.body_world.numpy(), device=self.device, dtype=torch.long)
        self._primary_body_ids = torch.nonzero(primary_body_mask, as_tuple=False).reshape(-1)
        self._body_slot_lookup = self._build_body_slot_lookup()
        body_com = getattr(model, "body_com", None)
        if body_com is None:
            self._body_com = torch.zeros((int(model.body_count), 3), dtype=torch.float32, device=self.device)
        else:
            self._body_com = _as_torch_array(body_com, self.device).reshape(-1, 3).to(dtype=torch.float32)
            if self._body_com.shape[0] != int(model.body_count):
                raise ValueError("model.body_com must provide one offset per body.")
        self._overflow: torch.Tensor | None = None
        self._body_ids: torch.Tensor | None = None
        self._overflow_total = 0
        self._overflow_total_gpu = torch.zeros((), dtype=torch.long, device=self.device)
        self._frames = 0
        self._row_capacity = 0
        self._row_buffers: dict[str, torch.Tensor] = {}

    @property
    def last_overflow(self) -> torch.Tensor | None:
        """Per-environment overflow count from the last ``encode`` call."""
        return self._overflow

    @property
    def last_body_ids(self) -> torch.Tensor | None:
        """Global owner-body ids for the last packed token frame."""
        return self._body_ids

    def reset_overflow_stats(self) -> None:
        """Reset cumulative overflow telemetry."""
        self._overflow_total = 0
        self._overflow_total_gpu.zero_()
        self._frames = 0

    def record_frame(self) -> None:
        """Increment the number of encoded frames for telemetry."""
        self._frames += 1

    def overflow_summary(self) -> dict[str, int | float]:
        """Return cumulative token overflow statistics."""
        return {
            "frames": self._frames,
            "overflow_tokens_total": self._overflow_total + int(self._overflow_total_gpu.item()),
            "overflow_frames": int((self._overflow > 0).sum().item()) if self._overflow is not None else 0,
        }

    def encode(
        self,
        raw_contacts: dict[str, torch.Tensor],
        state: newton.State,
    ) -> torch.Tensor:
        """Pack raw Newton contacts into ``[num_envs, max_contact_tokens, 17]`` tokens."""
        packed = torch.zeros(
            (self.num_envs, self.max_contact_tokens, CONTACT_TOKEN_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        overflow = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        body_ids = torch.full(
            (self.num_envs, self.max_contact_tokens),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._encode_into(raw_contacts, state, packed, overflow, body_ids)
        return packed

    def _encode_into(
        self,
        raw_contacts: dict[str, torch.Tensor],
        state: newton.State,
        packed: torch.Tensor,
        overflow: torch.Tensor,
        body_ids: torch.Tensor,
    ) -> None:
        """Pack contacts directly into caller-owned output buffers."""
        rows = self._directed_rows(raw_contacts, state)
        self._pack_rows(rows, packed, overflow, body_ids)
        self._overflow = overflow
        self._body_ids = body_ids
        self._overflow_total_gpu += overflow.sum()

    def _build_body_slot_lookup(self) -> torch.Tensor:
        """Map global body ids to per-environment local primary-body slots."""
        lookup = torch.full((int(self.model.body_count),), -1, dtype=torch.long, device=self.device)
        if self.primary_body_ids_by_world is not None:
            for env_ids in self.primary_body_ids_by_world:
                for local_slot, body_id in enumerate(env_ids.tolist()):
                    lookup[int(body_id)] = local_slot
            return lookup
        if self.body_world is not None:
            for world_id in range(self.num_envs):
                env_ids = torch.nonzero(
                    (self.body_world == world_id) & self.primary_body_mask.to(self.device),
                    as_tuple=False,
                ).reshape(-1)
                for local_slot, body_id in enumerate(env_ids.tolist()):
                    lookup[int(body_id)] = local_slot
            return lookup

        for world_id in range(self.num_envs):
            start = world_id * self.bodies_per_env
            end = start + self.bodies_per_env
            env_ids = [body_id for body_id in self._primary_body_ids.tolist() if start <= int(body_id) < end]
            for local_slot, body_id in enumerate(env_ids):
                lookup[int(body_id)] = local_slot
        return lookup

    def body_slot(self, body_ids: torch.Tensor) -> torch.Tensor:
        """Map Newton body IDs to per-world primary-body slots."""
        safe = body_ids.clamp(min=0, max=self._body_slot_lookup.shape[0] - 1)
        slots = self._body_slot_lookup.index_select(0, safe)
        return torch.where(body_ids >= 0, slots, torch.full_like(slots, -1))

    _body_slot = body_slot

    def directed_rows(self, raw: dict[str, torch.Tensor], state: newton.State) -> _DirectedContactRows:
        """Build the directed per-body rows used by native-pair and token encoders."""
        count = raw["shape0"].shape[0]
        if count == 0:
            empty = torch.empty(0, device=self.device)
            return _DirectedContactRows(
                valid=empty.bool(),
                world_id=empty.long(),
                body_id=empty.long(),
                other_body_id=empty.long(),
                other_dynamic=empty.bool(),
                point_world=empty.reshape(0, 3),
                normal_world=empty.reshape(0, 3),
                lever_world=empty.reshape(0, 3),
                gap=empty,
                velocity_world=empty.reshape(0, 3),
            )

        shape0 = raw["shape0"]
        shape1 = raw["shape1"]
        body0 = self._shape_body_ids(shape0)
        body1 = self._shape_body_ids(shape1)
        dynamic0 = body0 >= 0
        dynamic1 = body1 >= 0
        # Prefer effective surface points (support + offset) when available.
        point0 = raw.get("surface0_world", raw["point0_world"])
        point1 = raw.get("surface1_world", raw["point1_world"])
        normal01 = raw["normal"]
        gap = raw["surface_separation"]

        if "_body_q" in raw:
            body_q = raw["_body_q"]
        else:
            if state.body_q is None:
                raise ValueError("Contact encoding requires Newton body poses.")
            body_q = _as_torch_array(state.body_q, self.device)
        if "_body_qd" in raw:
            body_qd = raw["_body_qd"]
        else:
            if state.body_qd is None:
                raise ValueError("Contact encoding requires Newton body velocities.")
            body_qd = _as_torch_array(state.body_qd, self.device)

        com0 = self._body_com_world(body_q, body0)
        com1 = self._body_com_world(body_q, body1)
        vel0 = self._surface_velocity(dynamic0, body0, body_qd, point0, com0)
        vel1 = self._surface_velocity(dynamic1, body1, body_qd, point1, com1)

        world0 = self._contact_env_ids(body0, shape0)
        world1 = self._contact_env_ids(body1, shape1)
        # Prefer the dynamic robot side's env. Shared terrain shapes must not
        # pull contacts into env 0 via shape_id // bodies_per_env.
        pair_world = torch.where(
            dynamic0,
            world0,
            torch.where(dynamic1, world1, torch.full_like(world0, -1)),
        )

        valid0 = dynamic0 & (self._body_slot(body0) >= 0) & (pair_world >= 0)
        valid1 = dynamic1 & (self._body_slot(body1) >= 0) & (pair_world >= 0)

        rows = self._directed_row_scratch(count)
        first = slice(0, count)
        second = slice(count, 2 * count)
        for destination, first_value, second_value in (
            (rows.valid, valid0, valid1),
            (rows.world_id, pair_world, pair_world),
            (rows.body_id, body0, body1),
            (rows.other_body_id, body1, body0),
            (rows.other_dynamic, dynamic1, dynamic0),
            (rows.point_world, point0, point1),
            (rows.normal_world, normal01, -normal01),
            (rows.lever_world, point0 - com0, point1 - com1),
            (rows.gap, gap, gap),
            (rows.velocity_world, vel0 - vel1, vel1 - vel0),
        ):
            destination[first].copy_(first_value)
            destination[second].copy_(second_value)
        return rows

    _directed_rows = directed_rows

    def _directed_row_scratch(self, contact_count: int) -> _DirectedContactRows:
        """Return reusable buffers for two directed rows per contact."""
        required = 2 * contact_count
        if required > self._row_capacity:
            self._row_capacity = max(required, max(16, 2 * self._row_capacity))
            vector_shape = (self._row_capacity, 3)
            with torch.inference_mode(False):
                self._row_buffers = {
                    "valid": torch.empty(self._row_capacity, dtype=torch.bool, device=self.device),
                    "world_id": torch.empty(self._row_capacity, dtype=torch.long, device=self.device),
                    "body_id": torch.empty(self._row_capacity, dtype=torch.long, device=self.device),
                    "other_body_id": torch.empty(self._row_capacity, dtype=torch.long, device=self.device),
                    "other_dynamic": torch.empty(self._row_capacity, dtype=torch.bool, device=self.device),
                    "point_world": torch.empty(vector_shape, dtype=torch.float32, device=self.device),
                    "normal_world": torch.empty(vector_shape, dtype=torch.float32, device=self.device),
                    "lever_world": torch.empty(vector_shape, dtype=torch.float32, device=self.device),
                    "gap": torch.empty(self._row_capacity, dtype=torch.float32, device=self.device),
                    "velocity_world": torch.empty(vector_shape, dtype=torch.float32, device=self.device),
                }
        buffers = self._row_buffers
        return _DirectedContactRows(
            valid=buffers["valid"][:required],
            world_id=buffers["world_id"][:required],
            body_id=buffers["body_id"][:required],
            other_body_id=buffers["other_body_id"][:required],
            other_dynamic=buffers["other_dynamic"][:required],
            point_world=buffers["point_world"][:required],
            normal_world=buffers["normal_world"][:required],
            lever_world=buffers["lever_world"][:required],
            gap=buffers["gap"][:required],
            velocity_world=buffers["velocity_world"][:required],
        )

    def _pack_rows(
        self,
        rows: _DirectedContactRows,
        packed: torch.Tensor | None = None,
        overflow: torch.Tensor | None = None,
        packed_body_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack directed rows through the shared native-pair projector."""
        worlds = self.num_envs
        capacity = self.max_contact_tokens
        device = self.device
        packed_out = (
            torch.zeros((worlds, capacity, CONTACT_TOKEN_DIM), dtype=torch.float32, device=device)
            if packed is None
            else packed
        )
        overflow_out = torch.zeros(worlds, dtype=torch.long, device=device) if overflow is None else overflow
        body_ids_out = (
            torch.full((worlds, capacity), -1, dtype=torch.long, device=device)
            if packed_body_ids is None
            else packed_body_ids
        )
        if rows.valid.shape[0] % 2 != 0:
            raise RuntimeError("Directed contact rows must contain an even number of sides.")
        pairs = rows.valid.shape[0] // 2
        first = slice(0, pairs)
        second = slice(pairs, 2 * pairs)
        valid0 = rows.valid[:pairs]
        valid1 = rows.valid[pairs:]
        invalid_slot = torch.full((pairs,), -1, dtype=torch.long, device=device)
        native_pairs = {
            "body_ids_0": rows.body_id[first],
            "body_ids_1": rows.body_id[second],
            "body_slots_0": torch.where(valid0, self._body_slot(rows.body_id[first]), invalid_slot),
            "body_slots_1": torch.where(valid1, self._body_slot(rows.body_id[second]), invalid_slot),
            "world_ids": rows.world_id[first],
            "surface_points_0_w": rows.point_world[first],
            "surface_points_1_w": rows.point_world[second],
            "normal_w": rows.normal_world[first],
            "separation": rows.gap[first],
            "lever_arms_0_w": rows.lever_world[first],
            "lever_arms_1_w": rows.lever_world[second],
            "relative_velocities_0_w": rows.velocity_world[first],
            "relative_velocities_1_w": rows.velocity_world[second],
        }
        projected = project_native_pairs(
            native_pairs,
            num_frames=worlds,
            representation=CONTACT_REPRESENTATION_TOKENS,
            packing_policy="body_round_robin_pair_atomic",
            max_contact_tokens=capacity,
        )
        packed_out.copy_(projected["contact_tokens"])
        overflow_out.copy_(projected["contact_token_overflow"])
        body_ids_out.copy_(projected["contact_token_body_ids"])
        return packed_out, overflow_out, body_ids_out

    def _shape_body_ids(self, shapes: torch.Tensor) -> torch.Tensor:
        return gather_or_fill(self.shape_body_torch, shapes, -1)

    def _contact_env_ids(self, body_ids: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
        """Return per-contact env ids; ``-1`` marks unknown/static-only contacts."""
        dynamic = body_ids >= 0
        env_ids = torch.full(body_ids.shape, -1, dtype=torch.long, device=body_ids.device)
        if self.body_world is not None:
            body_world = self.body_world.to(device=body_ids.device, dtype=torch.long)
            env_ids = torch.where(dynamic, gather_or_fill(body_world, body_ids, -1), env_ids)
        static_envs = torch.div(shapes.long(), self.bodies_per_env, rounding_mode="floor")
        invalid_static = (shapes < 0) | (static_envs < 0) | (static_envs >= self.num_envs)
        static_envs = torch.where(invalid_static, torch.full_like(static_envs, -1), static_envs)
        env_ids = torch.where(dynamic, env_ids, static_envs)
        # Keep unknown env ids as -1 so packing can dump them instead of env 0.
        return env_ids

    def _body_com_world(self, body_q: torch.Tensor, body_ids: torch.Tensor) -> torch.Tensor:
        """Return world-frame body COM using ``model.body_com`` offsets."""
        safe = body_ids.clamp(min=0, max=body_q.shape[0] - 1)
        pose = body_q.index_select(0, safe)
        local_com = self._body_com.index_select(0, safe)
        return pose[:, :3] + torch_utils.quat_rotate(pose[:, 3:7], local_com)

    @staticmethod
    def _body_origin(body_q: torch.Tensor, body_ids: torch.Tensor) -> torch.Tensor:
        safe = body_ids.clamp(min=0, max=body_q.shape[0] - 1)
        return body_q.index_select(0, safe)[:, :3]

    @staticmethod
    def _surface_velocity(
        dynamic: torch.Tensor,
        body_ids: torch.Tensor,
        body_qd: torch.Tensor,
        points_world: torch.Tensor,
        com_world: torch.Tensor,
    ) -> torch.Tensor:
        safe = body_ids.clamp(min=0, max=body_qd.shape[0] - 1)
        velocity = body_qd.index_select(0, safe)
        value = velocity[:, :3] + torch.cross(velocity[:, 3:6], points_world - com_world, dim=-1)
        return torch.where(dynamic.unsqueeze(-1), value, torch.zeros_like(value))


def gather_owner_body_q(primary_body_q: torch.Tensor, contact_tokens: torch.Tensor) -> torch.Tensor:
    """Gather per-token owner poses from per-primary-body poses.

    Args:
        primary_body_q: Primary-body poses ``[B, T, P, 7]`` or ``[B, P, 7]``.
        contact_tokens: Packed tokens ``[B, T, K, D]`` or ``[B, K, D]``.

    Returns:
        Owner poses matching the token layout, shape ``[..., 7]``.
    """
    original_token_ndim = contact_tokens.ndim
    if contact_tokens.ndim == 3:
        contact_tokens = contact_tokens.unsqueeze(1)
    if primary_body_q.ndim == 3:
        primary_body_q = primary_body_q.unsqueeze(1)
    if primary_body_q.ndim != 4 or primary_body_q.shape[-1] != 7:
        raise ValueError(f"primary_body_q must have shape [B, T, P, 7], got {tuple(primary_body_q.shape)}.")
    if contact_tokens.shape[:2] != primary_body_q.shape[:2]:
        raise ValueError(
            "primary_body_q batch/time must match contact_tokens, "
            f"got poses {tuple(primary_body_q.shape)} and tokens {tuple(contact_tokens.shape)}."
        )
    num_primary = int(primary_body_q.shape[2])
    if num_primary <= 0:
        raise ValueError("primary_body_q must contain at least one primary body.")
    slots = contact_tokens[..., CONTACT_TOKEN_BODY_SLOT_INDEX].to(dtype=torch.long)
    safe = slots.clamp(min=0, max=num_primary - 1)
    gathered = torch.gather(primary_body_q, 2, safe.unsqueeze(-1).expand(*safe.shape, 7))
    if original_token_ndim == 3:
        return gathered.squeeze(1)
    return gathered


def transform_contact_tokens_to_body_frame(
    contact_tokens: torch.Tensor,
    root_body_q: torch.Tensor,
    *,
    translation_only: bool = False,
) -> torch.Tensor:
    """Transform token geometry channels from world frame into a body frame.

    ``root_body_q`` may be a shared pose per frame ``[B, T, 7]`` or a per-token
    pose ``[B, T, K, 7]``.
    """
    if contact_tokens is None:
        return contact_tokens
    if contact_tokens.numel() == 0:
        return contact_tokens

    original_shape = contact_tokens.shape
    if contact_tokens.ndim == 2:
        contact_tokens = contact_tokens.unsqueeze(0)
        root_body_q = root_body_q.unsqueeze(0)
    if contact_tokens.ndim == 3:
        contact_tokens = contact_tokens.unsqueeze(1)
        if (
            root_body_q.ndim == 2
            or root_body_q.ndim == 3
            and root_body_q.shape[-1] == 7
            and root_body_q.shape[1] == contact_tokens.shape[2]
        ):
            root_body_q = root_body_q.unsqueeze(1)

    batch, time, tokens, _ = contact_tokens.shape
    valid = contact_tokens[..., CONTACT_TOKEN_VALID_INDEX] > 0.5
    flat_tokens = contact_tokens.reshape(batch * time * tokens, CONTACT_TOKEN_DIM)
    flat_valid = valid.reshape(-1)
    if root_body_q.ndim == 4:
        if root_body_q.shape[:3] != (batch, time, tokens) or root_body_q.shape[-1] != 7:
            raise ValueError(
                "Per-token body poses must have shape [B, T, K, 7] matching contact_tokens, "
                f"got {tuple(root_body_q.shape)} for tokens {tuple(contact_tokens.shape)}."
            )
        flat_root = root_body_q.reshape(batch * time * tokens, 7)
    else:
        if root_body_q.shape[:2] != (batch, time) or root_body_q.shape[-1] != 7:
            raise ValueError(
                "Shared body poses must have shape [B, T, 7], "
                f"got {tuple(root_body_q.shape)} for tokens {tuple(contact_tokens.shape)}."
            )
        flat_root = root_body_q.reshape(batch * time, 7).repeat_interleave(tokens, dim=0)

    geometry = flat_tokens[:, CONTACT_TOKEN_GEOMETRY_SLICE].reshape(-1, 13)
    points = geometry[:, :3]
    normals = geometry[:, 3:6]
    levers = geometry[:, 6:9]
    gap = geometry[:, 9:10]
    rel_vel = geometry[:, 10:13]

    pos = flat_root[:, :3]
    quat = flat_root[:, 3:7]
    if translation_only:
        points_body = points - pos
        normals_body = normals
        levers_body = levers
        rel_vel_body = rel_vel
    else:
        points_body = torch_utils.transform_point_inverse(pos, quat, points)
        normals_body = torch_utils.quat_rotate_inverse(quat, normals)
        levers_body = torch_utils.quat_rotate_inverse(quat, levers)
        rel_vel_body = torch_utils.quat_rotate_inverse(quat, rel_vel)

    transformed = torch.cat(
        (
            flat_tokens[:, :4],
            points_body,
            normals_body,
            levers_body,
            gap,
            rel_vel_body,
        ),
        dim=-1,
    )
    transformed.mul_(flat_valid.unsqueeze(-1))
    result = transformed.reshape(batch, time, tokens, CONTACT_TOKEN_DIM)
    if len(original_shape) == 2:
        return result.reshape(original_shape)
    if len(original_shape) == 3:
        return result.squeeze(1)
    return result


def contact_token_validity_mask(contact_tokens: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask for valid contact tokens."""
    return contact_tokens[..., CONTACT_TOKEN_VALID_INDEX] > 0.5


def mask_invalid_contact_tokens(contact_tokens: torch.Tensor) -> torch.Tensor:
    """Return contact tokens with invalid rows zeroed."""
    valid = contact_token_validity_mask(contact_tokens).unsqueeze(-1)
    return torch.where(valid, contact_tokens, 0.0)
