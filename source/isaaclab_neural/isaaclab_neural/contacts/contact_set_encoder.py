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
    CONTACT_TOKEN_DIM,
    CONTACT_TOKEN_GAP_INDEX,
    CONTACT_TOKEN_GEOMETRY_SLICE,
    CONTACT_TOKEN_VALID_INDEX,
)
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
    solver_active: torch.Tensor
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
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys.index_select(0, order)
    rank_sorted = torch.zeros_like(sorted_keys)
    if sorted_keys.numel() == 0:
        return rank_sorted
    changes = torch.ones_like(sorted_keys, dtype=torch.bool)
    changes[1:] = sorted_keys[1:] != sorted_keys[:-1]
    segment_ids = torch.cumsum(changes.to(torch.long), dim=0) - 1
    counts = torch.bincount(segment_ids, minlength=int(segment_ids.max().item()) + 1)
    offsets = torch.zeros_like(counts)
    if counts.numel() > 1:
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)
    rank_sorted = torch.arange(sorted_keys.numel(), device=keys.device) - offsets.index_select(0, segment_ids)
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
    exclusive = torch.zeros_like(sorted_values)
    if sorted_values.numel() > 1:
        same = sorted_ids[1:] == sorted_ids[:-1]
        run_values = sorted_values.clone()
        for index in range(1, sorted_values.numel()):
            if same[index - 1]:
                run_values[index] = run_values[index - 1] + sorted_values[index]
            else:
                run_values[index] = sorted_values[index]
        exclusive[1:] = torch.where(same, run_values[:-1], 0.0)
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
        self._solver_active: torch.Tensor | None = None
        self._overflow_total = 0
        self._frames = 0

    @property
    def last_overflow(self) -> torch.Tensor | None:
        """Per-environment overflow count from the last ``encode`` call."""
        return self._overflow

    @property
    def last_body_ids(self) -> torch.Tensor | None:
        """Global owner-body ids for the last packed token frame."""
        return self._body_ids

    @property
    def last_solver_active(self) -> torch.Tensor | None:
        """Solver-active flags aligned with the last packed token frame."""
        return self._solver_active

    def reset_overflow_stats(self) -> None:
        """Reset cumulative overflow telemetry."""
        self._overflow_total = 0
        self._frames = 0

    def record_frame(self) -> None:
        """Increment the number of encoded frames for telemetry."""
        self._frames += 1

    def overflow_summary(self) -> dict[str, int | float]:
        """Return cumulative token overflow statistics."""
        return {
            "frames": self._frames,
            "overflow_tokens_total": self._overflow_total,
            "overflow_frames": int((self._overflow > 0).sum().item()) if self._overflow is not None else 0,
        }

    def encode(
        self,
        raw_contacts: dict[str, torch.Tensor],
        state: newton.State,
    ) -> torch.Tensor:
        """Pack raw Newton contacts into ``[num_envs, max_contact_tokens, 17]`` tokens."""
        rows = self._directed_rows(raw_contacts, state)
        packed, overflow, body_ids, solver_active = self._pack_rows(rows)
        if self._overflow is None or self._overflow.device != packed.device:
            self._overflow = torch.zeros(self.num_envs, dtype=torch.long, device=packed.device)
        self._overflow.copy_(overflow)
        self._body_ids = body_ids
        self._solver_active = solver_active
        self._overflow_total += int(overflow.sum().item())
        return packed

    def _build_body_slot_lookup(self) -> torch.Tensor:
        """Map global body ids to per-environment local primary-body slots."""
        lookup = torch.full((int(self.model.body_count),), -1, dtype=torch.long, device=self.device)
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

    def _body_slot(self, body_ids: torch.Tensor) -> torch.Tensor:
        safe = body_ids.clamp(min=0, max=self._body_slot_lookup.shape[0] - 1)
        slots = self._body_slot_lookup.index_select(0, safe)
        return torch.where(body_ids >= 0, slots, torch.full_like(slots, -1))

    def _directed_rows(self, raw: dict[str, torch.Tensor], state: newton.State) -> _DirectedContactRows:
        count = raw["shape0"].shape[0]
        if count == 0:
            empty = torch.empty(0, device=self.device)
            return _DirectedContactRows(
                valid=empty.bool(),
                solver_active=empty.bool(),
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
        solver_active = self._solver_active_mask(raw)

        body_q = _as_torch_array(state.body_q, self.device)
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

        rows0 = _DirectedContactRows(
            valid=valid0,
            solver_active=solver_active,
            world_id=pair_world,
            body_id=body0,
            other_body_id=body1,
            other_dynamic=dynamic1,
            point_world=point0,
            normal_world=normal01,
            lever_world=point0 - com0,
            gap=gap,
            velocity_world=vel0 - vel1,
        )
        rows1 = _DirectedContactRows(
            valid=valid1,
            solver_active=solver_active,
            world_id=pair_world,
            body_id=body1,
            other_body_id=body0,
            other_dynamic=dynamic0,
            point_world=point1,
            normal_world=-normal01,
            lever_world=point1 - com1,
            gap=gap,
            velocity_world=vel1 - vel0,
        )
        return self._concat_rows(rows0, rows1)

    @staticmethod
    def _concat_rows(first: _DirectedContactRows, second: _DirectedContactRows) -> _DirectedContactRows:
        return _DirectedContactRows(
            valid=torch.cat((first.valid, second.valid)),
            solver_active=torch.cat((first.solver_active, second.solver_active)),
            world_id=torch.cat((first.world_id, second.world_id)),
            body_id=torch.cat((first.body_id, second.body_id)),
            other_body_id=torch.cat((first.other_body_id, second.other_body_id)),
            other_dynamic=torch.cat((first.other_dynamic, second.other_dynamic)),
            point_world=torch.cat((first.point_world, second.point_world)),
            normal_world=torch.cat((first.normal_world, second.normal_world)),
            lever_world=torch.cat((first.lever_world, second.lever_world)),
            gap=torch.cat((first.gap, second.gap)),
            velocity_world=torch.cat((first.velocity_world, second.velocity_world)),
        )

    def _pack_rows(self, rows: _DirectedContactRows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        worlds = self.num_envs
        capacity = self.max_contact_tokens
        device = self.device
        packed = torch.zeros((worlds, capacity, CONTACT_TOKEN_DIM), dtype=torch.float32, device=device)
        overflow = torch.zeros(worlds, dtype=torch.long, device=device)
        if rows.valid.numel() == 0:
            body_ids = torch.full((worlds, capacity), -1, dtype=torch.long, device=device)
            solver_active = torch.zeros((worlds, capacity), dtype=torch.bool, device=device)
            return packed, overflow, body_ids, solver_active

        pairs = rows.valid.shape[0] // 2
        if rows.valid.shape[0] % 2 != 0:
            raise RuntimeError("Directed contact rows must contain an even number of sides.")

        valid0 = rows.valid[:pairs]
        valid1 = rows.valid[pairs:]
        pair_valid = valid0 | valid1
        sides = valid0.long() + valid1.long()
        # Unknown / invalid env ids are routed to dump world ``worlds``.
        pair_world = torch.where(
            (rows.world_id[:pairs] >= 0) & (rows.world_id[:pairs] < worlds),
            rows.world_id[:pairs],
            torch.full_like(rows.world_id[:pairs], worlds),
        )
        pair_gap = torch.where(pair_valid, rows.gap[:pairs], torch.full_like(rows.gap[:pairs], float("inf")))

        body_key = torch.where(
            rows.valid,
            rows.body_id,
            torch.full_like(rows.body_id, worlds * self.bodies_per_env),
        )
        gap_key = torch.where(rows.valid, rows.gap, torch.full_like(rows.gap, float("inf")))
        order = torch.argsort(gap_key, stable=True)
        order = order.index_select(0, torch.argsort(body_key.index_select(0, order), stable=True))
        rank = torch.empty_like(rows.gap, dtype=torch.long)
        rank.index_copy_(0, order, _segment_rank(body_key.index_select(0, order), rows.gap.new_zeros(rows.gap.shape)))
        rank = torch.where(rows.valid, rank, torch.full_like(rank, 2 * pairs))

        pair_rank = torch.minimum(rank[:pairs], rank[pairs:])
        pair_order = torch.argsort(pair_gap, stable=True)
        pair_order = pair_order.index_select(0, torch.argsort(pair_rank.index_select(0, pair_order), stable=True))
        pair_order = pair_order.index_select(0, torch.argsort(pair_world.index_select(0, pair_order), stable=True))

        world_sorted = pair_world.index_select(0, pair_order)
        sides_sorted = sides.index_select(0, pair_order)
        offset_sorted = _segment_exclusive_sum(world_sorted, sides_sorted, sides.new_zeros(sides.shape))
        keep_sorted = (world_sorted < worlds) & (offset_sorted + sides_sorted <= capacity)
        offset = torch.empty_like(offset_sorted)
        offset.index_copy_(0, pair_order, offset_sorted)
        keep_pair = torch.zeros_like(keep_sorted)
        keep_pair.index_copy_(0, pair_order, keep_sorted)

        keep_row = torch.cat((keep_pair, keep_pair)) & rows.valid
        within = torch.cat((torch.zeros_like(valid0, dtype=torch.long), valid0.long()))
        # Rejected/invalid rows write into one extra dump row so they cannot
        # overwrite the next environment's first packed slot.
        dump_row = worlds * capacity
        world_row = torch.where(
            (rows.world_id >= 0) & (rows.world_id < worlds),
            rows.world_id,
            torch.full_like(rows.world_id, worlds),
        )
        destination = torch.where(
            keep_row,
            world_row * capacity + torch.cat((offset, offset)) + within,
            dump_row,
        )

        body_slot = self._body_slot(rows.body_id)
        other_slot = self._body_slot(rows.other_body_id)
        other_slot = torch.where(rows.other_dynamic, other_slot, torch.full_like(other_slot, -1))

        candidates = torch.stack(
            (
                rows.valid.to(torch.float32),
                body_slot.to(torch.float32),
                other_slot.to(torch.float32),
                rows.other_dynamic.to(torch.float32),
                rows.point_world[:, 0],
                rows.point_world[:, 1],
                rows.point_world[:, 2],
                rows.normal_world[:, 0],
                rows.normal_world[:, 1],
                rows.normal_world[:, 2],
                rows.lever_world[:, 0],
                rows.lever_world[:, 1],
                rows.lever_world[:, 2],
                rows.gap,
                rows.velocity_world[:, 0],
                rows.velocity_world[:, 1],
                rows.velocity_world[:, 2],
            ),
            dim=-1,
        )
        flat = candidates.new_zeros((dump_row + 1, CONTACT_TOKEN_DIM))
        flat.index_copy_(0, destination, candidates)
        packed = flat[:dump_row].reshape(worlds, capacity, CONTACT_TOKEN_DIM)
        flat_body_ids = torch.full((dump_row + 1,), -1, dtype=torch.long, device=device)
        flat_body_ids.index_copy_(0, destination, rows.body_id)
        packed_body_ids = flat_body_ids[:dump_row].reshape(worlds, capacity)
        packed[..., CONTACT_TOKEN_VALID_INDEX] = (packed[..., CONTACT_TOKEN_VALID_INDEX] > 0.5).to(torch.float32)
        flat_solver_active = torch.zeros(dump_row + 1, dtype=torch.bool, device=device)
        flat_solver_active.index_copy_(0, destination, rows.solver_active)
        packed_solver_active = flat_solver_active[:dump_row].reshape(worlds, capacity)
        packed_solver_active &= packed[..., CONTACT_TOKEN_VALID_INDEX] > 0.5

        dropped = rows.valid & ~keep_row
        if dropped.any():
            dropped_counts = torch.zeros(worlds + 1, dtype=torch.long, device=device)
            dropped_counts.scatter_add_(0, world_row[dropped], torch.ones_like(world_row[dropped]))
            overflow.copy_(dropped_counts[:worlds])
        return packed, overflow, packed_body_ids, packed_solver_active

    @staticmethod
    def _solver_active_mask(raw: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return Newton's strict solver-inclusion decision for each raw contact."""
        required = ("point0_world", "point1_world", "normal", "thickness0", "thickness1")
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError(f"Solver-active contact classification requires raw fields: {missing}.")
        clearance = torch.sum(raw["normal"] * (raw["point1_world"] - raw["point0_world"]), dim=-1)
        clearance = clearance - raw["thickness0"] - raw["thickness1"]
        return clearance < 0.0

    def _shape_body_ids(self, shapes: torch.Tensor) -> torch.Tensor:
        body_ids = torch.full(shapes.shape, -1, dtype=torch.long, device=shapes.device)
        valid_shapes = (shapes >= 0) & (shapes < self.shape_body_torch.shape[0])
        body_ids[valid_shapes] = self.shape_body_torch[shapes[valid_shapes].long()]
        return body_ids

    def _contact_env_ids(self, body_ids: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
        """Return per-contact env ids; ``-1`` marks unknown/static-only contacts."""
        env_ids = torch.full(body_ids.shape, -1, dtype=torch.long, device=body_ids.device)
        dynamic = body_ids >= 0
        if dynamic.any() and self.body_world is not None:
            env_ids[dynamic] = self.body_world.to(device=body_ids.device)[body_ids[dynamic].long()]
        static = ~dynamic
        if static.any():
            static_envs = torch.div(shapes[static].long(), self.bodies_per_env, rounding_mode="floor")
            invalid_static = (shapes[static] < 0) | (static_envs < 0) | (static_envs >= self.num_envs)
            env_ids[static] = torch.where(
                invalid_static,
                torch.full_like(static_envs, -1, dtype=torch.long),
                static_envs.to(dtype=torch.long),
            )
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


def transform_contact_tokens_to_body_frame(
    contact_tokens: torch.Tensor,
    root_body_q: torch.Tensor,
    *,
    translation_only: bool = False,
) -> torch.Tensor:
    """Transform token geometry channels from world frame into the root body frame."""
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
        root_body_q = root_body_q.unsqueeze(1)

    contact_tokens = contact_tokens.clone()
    batch, time, tokens, _ = contact_tokens.shape
    valid = contact_tokens[..., CONTACT_TOKEN_VALID_INDEX] > 0.5
    flat_tokens = contact_tokens.reshape(batch * time * tokens, CONTACT_TOKEN_DIM)
    flat_valid = valid.reshape(-1)
    flat_root = root_body_q.reshape(batch * time, 7).repeat_interleave(tokens, dim=0)

    if flat_valid.any():
        geometry = flat_tokens[flat_valid, CONTACT_TOKEN_GEOMETRY_SLICE].reshape(-1, 13)
        points = geometry[:, :3]
        normals = geometry[:, 3:6]
        levers = geometry[:, 6:9]
        gap = geometry[:, 9:10]
        rel_vel = geometry[:, 10:13]

        pos = flat_root[flat_valid, :3]
        quat = flat_root[flat_valid, 3:7]
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

        flat_tokens[flat_valid, 4:7] = points_body
        flat_tokens[flat_valid, 7:10] = normals_body
        flat_tokens[flat_valid, 10:13] = levers_body
        flat_tokens[flat_valid, CONTACT_TOKEN_GAP_INDEX : CONTACT_TOKEN_GAP_INDEX + 1] = gap
        flat_tokens[flat_valid, 14:17] = rel_vel_body

    flat_tokens[~flat_valid] = 0.0
    result = flat_tokens.reshape(batch, time, tokens, CONTACT_TOKEN_DIM)
    if len(original_shape) == 2:
        return result.reshape(original_shape)
    if len(original_shape) == 3:
        return result.squeeze(1)
    return result


def contact_token_validity_mask(contact_tokens: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask for valid contact tokens."""
    return contact_tokens[..., CONTACT_TOKEN_VALID_INDEX] > 0.5


def mask_invalid_contact_tokens(contact_tokens: torch.Tensor) -> torch.Tensor:
    """Zero invalid token rows in-place semantics via clone."""
    masked = contact_tokens.clone()
    valid = contact_token_validity_mask(masked).unsqueeze(-1)
    return torch.where(valid, masked, torch.zeros_like(masked))
