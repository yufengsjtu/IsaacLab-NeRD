# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Literal

import newton
import numpy as np
import torch
import warp as wp

from isaaclab_neural.contacts.active15_contact_encoder import Active15ContactEncoder
from isaaclab_neural.contacts.contact_set_encoder import ContactSetEncoder
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_FILTER_NONE,
    CONTACT_FILTER_SOLVER_ACTIVE,
    CONTACT_REPRESENTATION_ACTIVE15,
    CONTACT_REPRESENTATION_ACTIVE15_SELF,
    CONTACT_REPRESENTATION_FLAT,
    CONTACT_REPRESENTATION_RAW15,
    CONTACT_REPRESENTATION_RAW15_SELF,
    CONTACT_REPRESENTATION_TOKENS,
    CONTACT_TOKEN_DIM,
    CONTACT_TOKEN_SELF_COLLISION_FIELD,
    CONTACT_TOKEN_SOLVER_ACTIVE_FIELD,
    DEFAULT_MAX_CONTACT_TOKENS,
    is_contact_token_representation,
    is_native15_self_contact_representation,
)
from isaaclab_neural.contacts.native15_self_contact_encoder import (
    Active15SelfContactEncoder,
    Raw15SelfContactEncoder,
)
from isaaclab_neural.contacts.packing import ContactPackingPolicy, get_contact_order
from isaaclab_neural.contacts.raw15_contact_encoder import Raw15ContactEncoder
from isaaclab_neural.utils import torch_utils

logger = logging.getLogger(__name__)


class NewtonContactAdapter:
    """
    Adapter from Newton's native dynamic contacts to NeRD's fixed-size contact input.

    Newton's collision pipeline emits a variable number of contacts each frame.
    NeRD models expect a fixed `(num_envs, num_contacts_per_env, ...)` layout,
    so this class packs the native contacts into fixed buffers with masks.
    """

    def __init__(
        self,
        model: newton.Model,
        num_contacts_per_env: int,
        device: str | None = None,
        packing_policy: ContactPackingPolicy = "stable_index",
        *,
        contact_representation: str = CONTACT_REPRESENTATION_FLAT,
        contact_filter: str = CONTACT_FILTER_NONE,
        max_contact_tokens: int = DEFAULT_MAX_CONTACT_TOKENS,
    ):
        self.model = model
        self.num_envs = int(model.world_count)
        self.contact_representation = contact_representation
        if contact_filter not in {CONTACT_FILTER_NONE, CONTACT_FILTER_SOLVER_ACTIVE}:
            raise ValueError(f"Unsupported contact_filter: {contact_filter!r}.")
        if contact_filter != CONTACT_FILTER_NONE and not is_contact_token_representation(contact_representation):
            raise ValueError("contact_filter requires a contact-token representation.")
        self.contact_filter = contact_filter
        self.max_contact_tokens = int(max_contact_tokens)
        if is_contact_token_representation(self.contact_representation):
            self.num_contacts_per_env = self.max_contact_tokens
        else:
            self.num_contacts_per_env = int(num_contacts_per_env)
        self.num_total_contacts = self.num_envs * self.num_contacts_per_env
        self.packing_policy: ContactPackingPolicy = packing_policy

        if device is None:
            self.device = wp.device_to_torch(model.device)
        else:
            self.device = torch.device(device)

        if model.shape_body is None:
            raise ValueError("NewtonContactAdapter requires model.shape_body.")
        self.shape_body = model.shape_body.numpy()
        self.shape_body_torch = torch.as_tensor(self.shape_body, dtype=torch.long, device=self.device)
        self.primary_body_mask, self.contact_side_rule = self._derive_primary_body_mask()
        if int(model.body_count) % self.num_envs != 0:
            raise ValueError("NewtonContactAdapter requires a uniform body count across environments.")
        self.bodies_per_env = int(model.body_count // model.world_count)
        self.body_world = model.body_world.numpy() if model.body_world is not None else None
        self.body_world_torch = (
            torch.as_tensor(self.body_world, dtype=torch.long, device=self.device)
            if self.body_world is not None
            else None
        )
        if self.body_world_torch is None:
            primary_counts = self.primary_body_mask.reshape(self.num_envs, self.bodies_per_env).sum(dim=1)
        else:
            primary_counts = torch.stack(
                [self.primary_body_mask[self.body_world_torch == world_id].sum() for world_id in range(self.num_envs)]
            )
        if not torch.all(primary_counts == primary_counts[0]):
            raise ValueError("NewtonContactAdapter requires the same primary body count in every environment.")
        self.num_primary_bodies_per_env = int(primary_counts[0].item())
        body_local_ids = np.empty(int(model.body_count), dtype=np.int64)
        if self.body_world is None:
            body_local_ids[:] = np.arange(int(model.body_count)) % self.bodies_per_env
        else:
            for world_id in range(self.num_envs):
                world_bodies = np.flatnonzero(self.body_world == world_id)
                body_local_ids[world_bodies] = np.arange(world_bodies.size)
        self.body_local_ids = torch.as_tensor(body_local_ids, dtype=torch.long, device=self.device)
        self._contact_frames = 0
        self._raw_contacts_total = 0
        self._packed_contacts_total = 0
        self._dropped_contacts_total = 0
        self._truncated_frames = 0
        self._truncation_warning_emitted = False
        # Deferred GPU counters (synced only in truncation_summary / warnings).
        self._packed_contacts_gpu = torch.zeros((), dtype=torch.long, device=self.device)
        self._dropped_contacts_gpu = torch.zeros((), dtype=torch.long, device=self.device)
        self._truncated_frames_gpu = torch.zeros((), dtype=torch.long, device=self.device)
        self.contact_masks = torch.zeros(
            (self.num_envs, self.num_contacts_per_env),
            dtype=torch.bool,
            device=self.device,
        )
        self.contact_normals = torch.zeros(
            (self.num_envs, self.num_contacts_per_env, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.contact_depths = torch.zeros(
            (self.num_envs, self.num_contacts_per_env),
            dtype=torch.float32,
            device=self.device,
        )
        self.contact_body_ids_0 = torch.full(
            (self.num_envs, self.num_contacts_per_env),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.contact_body_ids_1 = torch.full(
            (self.num_envs, self.num_contacts_per_env),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.contact_thicknesses_0 = torch.zeros(
            (self.num_envs, self.num_contacts_per_env),
            dtype=torch.float32,
            device=self.device,
        )
        self.contact_thicknesses_1 = torch.zeros(
            (self.num_envs, self.num_contacts_per_env),
            dtype=torch.float32,
            device=self.device,
        )
        self.contact_points_0 = torch.zeros(
            (self.num_envs, self.num_contacts_per_env, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.contact_points_1 = torch.zeros(
            (self.num_envs, self.num_contacts_per_env, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.contact_tokens = torch.zeros(
            (self.num_envs, self.max_contact_tokens, CONTACT_TOKEN_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self.contact_token_overflow = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )
        self.contact_token_body_ids = torch.full(
            (self.num_envs, self.max_contact_tokens),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.contact_token_solver_active = torch.zeros(
            (self.num_envs, self.max_contact_tokens),
            dtype=torch.bool,
            device=self.device,
        )
        self.contact_token_self_collision = torch.zeros(
            (self.num_envs, self.max_contact_tokens),
            dtype=torch.bool,
            device=self.device,
        )
        self._token_encoder: ContactSetEncoder | None = None
        if is_contact_token_representation(self.contact_representation):
            encoder_types = {
                CONTACT_REPRESENTATION_ACTIVE15: Active15ContactEncoder,
                CONTACT_REPRESENTATION_RAW15: Raw15ContactEncoder,
                CONTACT_REPRESENTATION_ACTIVE15_SELF: Active15SelfContactEncoder,
                CONTACT_REPRESENTATION_RAW15_SELF: Raw15SelfContactEncoder,
            }
            filtered_raw_views = {
                CONTACT_REPRESENTATION_ACTIVE15: Raw15ContactEncoder,
                CONTACT_REPRESENTATION_ACTIVE15_SELF: Raw15SelfContactEncoder,
            }
            if (
                self.contact_representation in filtered_raw_views
                and self.contact_filter == CONTACT_FILTER_SOLVER_ACTIVE
            ):
                encoder_type = filtered_raw_views[self.contact_representation]
            else:
                encoder_type = encoder_types.get(self.contact_representation, ContactSetEncoder)
            self._token_encoder = encoder_type(
                model=model,
                primary_body_mask=self.primary_body_mask,
                shape_body_torch=self.shape_body_torch,
                bodies_per_env=self.bodies_per_env,
                num_envs=self.num_envs,
                max_contact_tokens=self.max_contact_tokens,
                device=self.device,
            )

    def reset_buffers(self) -> None:
        """Clear all fixed-size contact buffers before packing a new frame."""
        self.contact_masks.zero_()
        self.contact_normals.zero_()
        self.contact_depths.zero_()
        if hasattr(self, "contact_body_ids_0"):
            self.contact_body_ids_0.fill_(-1)
            self.contact_body_ids_1.fill_(-1)
        self.contact_thicknesses_0.zero_()
        self.contact_thicknesses_1.zero_()
        self.contact_points_0.zero_()
        self.contact_points_1.zero_()
        self.contact_tokens.zero_()
        self.contact_token_overflow.zero_()
        if hasattr(self, "contact_token_body_ids"):
            self.contact_token_body_ids.fill_(-1)
        if hasattr(self, "contact_token_solver_active"):
            self.contact_token_solver_active.zero_()
        if hasattr(self, "contact_token_self_collision"):
            self.contact_token_self_collision.zero_()

    def update(
        self,
        contacts: newton.Contacts,
        state: newton.State | None = None,
    ) -> None:
        """Pack Newton native contacts into fixed NeRD contact buffers."""
        self.reset_buffers()

        contact_count = self._contact_count(contacts)
        self._contact_frames += 1
        if contact_count == 0:
            return

        raw = self._read_raw_contacts(contacts, contact_count, state)
        if is_contact_token_representation(self.contact_representation):
            assert self._token_encoder is not None
            if state is None:
                raise ValueError("Contact token encoding requires the current Newton State.")
            self.contact_tokens.copy_(self._token_encoder.encode(raw, state))
            self._token_encoder.record_frame()
            overflow = self._token_encoder.last_overflow
            if overflow is not None:
                self.contact_token_overflow.copy_(overflow)
            body_ids = self._token_encoder.last_body_ids
            if body_ids is not None:
                self.contact_token_body_ids.copy_(body_ids)
            if is_native15_self_contact_representation(self.contact_representation):
                self_collision = self._token_encoder.last_self_collision
                if self_collision is None:
                    raise RuntimeError("Native15 self-collision encoding did not produce aligned flags.")
                self.contact_token_self_collision.copy_(self_collision)
            if self.contact_representation == CONTACT_REPRESENTATION_TOKENS:
                solver_active = self._token_encoder.last_solver_active
                if solver_active is None:
                    raise RuntimeError("Contact-token encoding did not produce solver-active flags.")
                self.contact_token_solver_active.copy_(solver_active)
            valid_tokens = (self.contact_tokens[..., 0] > 0.5).sum()
            overflow_sum = self.contact_token_overflow.sum()
            self._raw_contacts_total += contact_count
            self._packed_contacts_gpu += valid_tokens.to(dtype=torch.long)
            dropped = overflow_sum.to(dtype=torch.long)
            self._dropped_contacts_gpu += dropped
            self._truncated_frames_gpu += (dropped > 0).to(dtype=torch.long)
            if not self._truncation_warning_emitted and int(dropped.item()) > 0:
                logger.warning(
                    "Contact token packing dropped tokens because max_contact_tokens=%d. "
                    "Further truncation warnings are suppressed; inspect truncation_summary() for totals.",
                    self.max_contact_tokens,
                )
                self._truncation_warning_emitted = True
            return

        self._pack_flat_contacts(raw)

    def _pack_flat_contacts(self, raw: dict[str, torch.Tensor]) -> None:
        """Vectorized flat packing matching sequential penetration/slot semantics."""
        if not hasattr(self, "_packed_contacts_gpu"):
            self._packed_contacts_gpu = torch.zeros((), dtype=torch.long, device=self.device)
            self._dropped_contacts_gpu = torch.zeros((), dtype=torch.long, device=self.device)
            self._truncated_frames_gpu = torch.zeros((), dtype=torch.long, device=self.device)

        order = get_contact_order(raw, self.packing_policy)
        shape0 = raw["shape0"][order]
        shape1 = raw["shape1"][order]
        normals = raw["normal"][order]
        depths = raw["surface_separation"][order]
        thickness0 = raw["thickness0"][order]
        thickness1 = raw["thickness1"][order]
        points0 = raw["point0_world"][order]
        points1 = raw["point1_world"][order]

        env_ids = self._contact_env_ids(shape0, shape1)
        valid = (env_ids >= 0) & (env_ids < self.num_envs)
        frame_raw = int(valid.sum().item())
        self._raw_contacts_total += frame_raw
        if frame_raw == 0:
            return

        env_ids_v = env_ids[valid]
        n_valid = env_ids_v.shape[0]
        # Rank contacts within each env in encounter (priority) order.
        positions = torch.arange(n_valid, device=self.device, dtype=torch.long)
        sorted_env, sort_idx = torch.sort(env_ids_v, stable=True)
        is_start = torch.ones(n_valid, dtype=torch.bool, device=self.device)
        if n_valid > 1:
            is_start[1:] = sorted_env[1:] != sorted_env[:-1]
        group_start_ids = torch.cumsum(is_start.to(dtype=torch.long), dim=0) - 1
        start_pos = positions[is_start]
        rank_sorted = positions - start_pos[group_start_ids]
        rank = torch.empty_like(rank_sorted)
        rank[sort_idx] = rank_sorted

        keep = rank < self.num_contacts_per_env
        dropped = (~keep).sum()
        packed = keep.sum()
        self._packed_contacts_gpu += packed.to(dtype=torch.long)
        self._dropped_contacts_gpu += dropped.to(dtype=torch.long)
        self._truncated_frames_gpu += (dropped > 0).to(dtype=torch.long)

        if not self._truncation_warning_emitted and int(dropped.item()) > 0:
            logger.warning(
                "Newton-native contact packing dropped %d contacts because num_contacts_per_env=%d. "
                "Further truncation warnings are suppressed; inspect truncation_summary() for totals.",
                int(dropped.item()),
                self.num_contacts_per_env,
            )
            self._truncation_warning_emitted = True

        if not bool(keep.any()):
            return

        ordered_valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        keep_local = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        src = ordered_valid_idx[keep_local]
        env_keep = env_ids_v[keep_local]
        slot_keep = rank[keep_local]

        self.contact_masks[env_keep, slot_keep] = True
        self.contact_normals[env_keep, slot_keep] = normals[src]
        self.contact_depths[env_keep, slot_keep] = depths[src]
        if hasattr(self, "contact_body_ids_0"):
            body0 = self._shape_body_ids(shape0[src])
            body1 = self._shape_body_ids(shape1[src])
            valid_body0 = body0 >= 0
            valid_body1 = body1 >= 0
            self.contact_body_ids_0[env_keep[valid_body0], slot_keep[valid_body0]] = self.body_local_ids[
                body0[valid_body0]
            ]
            self.contact_body_ids_1[env_keep[valid_body1], slot_keep[valid_body1]] = self.body_local_ids[
                body1[valid_body1]
            ]
        self.contact_thicknesses_0[env_keep, slot_keep] = thickness0[src]
        self.contact_thicknesses_1[env_keep, slot_keep] = thickness1[src]
        self.contact_points_0[env_keep, slot_keep] = points0[src]
        self.contact_points_1[env_keep, slot_keep] = points1[src]

    def _pack_flat_contacts_sequential(self, raw: dict[str, torch.Tensor]) -> None:
        """Reference Python packing used by equivalence tests."""
        order = get_contact_order(raw, self.packing_policy)
        write_counts = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        frame_raw_contacts = 0
        frame_packed_contacts = 0
        frame_dropped_contacts = 0

        for contact_idx in order.tolist():
            shape0 = int(raw["shape0"][contact_idx].item())
            shape1 = int(raw["shape1"][contact_idx].item())
            env_id = self._contact_env_id(shape0, shape1)
            if env_id < 0 or env_id >= self.num_envs:
                continue

            frame_raw_contacts += 1
            slot_id = int(write_counts[env_id].item())
            if slot_id >= self.num_contacts_per_env:
                frame_dropped_contacts += 1
                continue

            self.contact_masks[env_id, slot_id] = True
            self.contact_normals[env_id, slot_id].copy_(raw["normal"][contact_idx])
            self.contact_depths[env_id, slot_id] = raw["surface_separation"][contact_idx]
            self.contact_thicknesses_0[env_id, slot_id] = raw["thickness0"][contact_idx]
            self.contact_thicknesses_1[env_id, slot_id] = raw["thickness1"][contact_idx]
            self.contact_points_0[env_id, slot_id].copy_(raw["point0_world"][contact_idx])
            self.contact_points_1[env_id, slot_id].copy_(raw["point1_world"][contact_idx])

            write_counts[env_id] += 1
            frame_packed_contacts += 1

        self._raw_contacts_total += frame_raw_contacts
        self._packed_contacts_total += frame_packed_contacts
        self._dropped_contacts_total += frame_dropped_contacts
        if frame_dropped_contacts > 0:
            self._truncated_frames += 1

    def to_neural_inputs(self) -> dict[str, torch.Tensor]:
        """Return contact tensors shaped like fixed-ground neural inputs.

        Returns views into the adapter buffers. Callers that retain contacts
        across frames (e.g. transformer history) must clone before the next
        :meth:`update`, which :class:`TransformerNeuralSolver` already does.
        """
        if is_contact_token_representation(self.contact_representation):
            inputs = {
                "contact_tokens": self.contact_tokens,
                "contact_token_overflow": self.contact_token_overflow,
            }
            if self.contact_representation == CONTACT_REPRESENTATION_TOKENS:
                inputs[CONTACT_TOKEN_SOLVER_ACTIVE_FIELD] = self.contact_token_solver_active
            if is_native15_self_contact_representation(self.contact_representation):
                inputs[CONTACT_TOKEN_SELF_COLLISION_FIELD] = self.contact_token_self_collision
            return inputs

        B = self.num_envs
        C = self.num_contacts_per_env
        return {
            "contact_masks": self.contact_masks,
            "contact_normals": self.contact_normals.reshape(B, C * 3),
            "contact_depths": self.contact_depths,
            "contact_thicknesses_0": self.contact_thicknesses_0,
            "contact_thicknesses_1": self.contact_thicknesses_1,
            "contact_points_0": self.contact_points_0.reshape(B, C * 3),
            "contact_points_1": self.contact_points_1.reshape(B, C * 3),
        }

    def truncation_summary(self) -> dict[str, int | float]:
        """Return cumulative native contact packing statistics."""
        packed_gpu = int(getattr(self, "_packed_contacts_gpu", torch.tensor(0)).item())
        dropped_gpu = int(getattr(self, "_dropped_contacts_gpu", torch.tensor(0)).item())
        truncated_gpu = int(getattr(self, "_truncated_frames_gpu", torch.tensor(0)).item())
        packed = self._packed_contacts_total + packed_gpu
        dropped = self._dropped_contacts_total + dropped_gpu
        truncated = self._truncated_frames + truncated_gpu
        summary = {
            "frames": self._contact_frames,
            "raw_contacts": self._raw_contacts_total,
            "packed_contacts": packed,
            "dropped_contacts": dropped,
            "truncated_frames": truncated,
            "truncated_frame_ratio": truncated / max(self._contact_frames, 1),
        }
        if self._token_encoder is not None:
            summary.update(self._token_encoder.overflow_summary())
        return summary

    def _contact_count(self, contacts: newton.Contacts) -> int:
        count_array = contacts.rigid_contact_count
        if count_array is None:
            return 0
        count = int(wp.to_torch(count_array)[0].item())  # type: ignore[arg-type]
        return min(count, int(contacts.rigid_contact_max))

    def _read_raw_contacts(
        self,
        contacts: newton.Contacts,
        contact_count: int,
        state: newton.State | None,
    ) -> dict[str, torch.Tensor]:
        shape0 = wp.to_torch(contacts.rigid_contact_shape0)[:contact_count].to(self.device)
        shape1 = wp.to_torch(contacts.rigid_contact_shape1)[:contact_count].to(self.device)
        point0 = wp.to_torch(contacts.rigid_contact_point0)[:contact_count].to(self.device)
        point1 = wp.to_torch(contacts.rigid_contact_point1)[:contact_count].to(self.device)
        normal = wp.to_torch(contacts.rigid_contact_normal)[:contact_count].to(self.device)

        thickness0 = self._read_thickness(contacts, contact_count, side=0)
        thickness1 = self._read_thickness(contacts, contact_count, side=1)
        offset0 = self._read_optional_vec3(contacts, "rigid_contact_offset0", contact_count)
        offset1 = self._read_optional_vec3(contacts, "rigid_contact_offset1", contact_count)
        if (offset0 is None) != (offset1 is None):
            zeros = torch.zeros_like(point0)
            offset0 = zeros if offset0 is None else offset0
            offset1 = zeros if offset1 is None else offset1
        shape0, shape1, point0, point1, normal, thickness0, thickness1, offset0, offset1 = (
            self._canonicalize_contact_sides(
                shape0,
                shape1,
                point0,
                point1,
                normal,
                thickness0,
                thickness1,
                offset0,
                offset1,
            )
        )
        surface_separation, point0_world, point1_world, surface0_world, surface1_world = (
            self._read_separation_and_world_points(
                shape0,
                shape1,
                point0,
                point1,
                normal,
                thickness0,
                thickness1,
                offset0,
                offset1,
                state,
            )
        )

        return {
            "shape0": shape0,
            "shape1": shape1,
            "point0": point0,
            "point1": point1,
            "point0_world": point0_world,
            "point1_world": point1_world,
            "surface0_world": surface0_world,
            "surface1_world": surface1_world,
            "normal": normal,
            "surface_separation": surface_separation,
            "thickness0": thickness0,
            "thickness1": thickness1,
        }

    def _canonicalize_contact_sides(
        self,
        shape0: torch.Tensor,
        shape1: torch.Tensor,
        point0: torch.Tensor,
        point1: torch.Tensor,
        normal: torch.Tensor,
        thickness0: torch.Tensor,
        thickness1: torch.Tensor,
        offset0: torch.Tensor | None = None,
        offset1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Put the dynamic robot side first and orient normals from side 0 to side 1."""
        body0 = self._shape_body_ids(shape0)
        body1 = self._shape_body_ids(shape1)
        primary0 = self._is_primary_body(body0)
        primary1 = self._is_primary_body(body1)
        dynamic0 = body0 >= 0
        dynamic1 = body1 >= 0

        # Prefer the primary articulation (the controlled robot) over other
        # dynamic bodies. Fall back to dynamic-first and then shape ordering.
        fallback_swap = (~dynamic0 & dynamic1) | ((dynamic0 == dynamic1) & (shape0 > shape1))
        swap = (~primary0 & primary1) | ((primary0 == primary1) & fallback_swap)
        vector_swap = swap.unsqueeze(-1)

        canonical_shape0 = torch.where(swap, shape1, shape0)
        canonical_shape1 = torch.where(swap, shape0, shape1)
        canonical_point0 = torch.where(vector_swap, point1, point0)
        canonical_point1 = torch.where(vector_swap, point0, point1)
        canonical_normal = torch.where(vector_swap, -normal, normal)
        canonical_thickness0 = torch.where(swap, thickness1, thickness0)
        canonical_thickness1 = torch.where(swap, thickness0, thickness1)
        canonical_offset0 = None if offset0 is None else torch.where(vector_swap, offset1, offset0)
        canonical_offset1 = None if offset1 is None else torch.where(vector_swap, offset0, offset1)
        return (
            canonical_shape0,
            canonical_shape1,
            canonical_point0,
            canonical_point1,
            canonical_normal,
            canonical_thickness0,
            canonical_thickness1,
            canonical_offset0,
            canonical_offset1,
        )

    def _shape_body_ids(self, shapes: torch.Tensor) -> torch.Tensor:
        """Return owning body indices, using ``-1`` for static or invalid shapes."""
        body_ids = torch.full(shapes.shape, -1, dtype=torch.long, device=shapes.device)
        valid_shapes = (shapes >= 0) & (shapes < self.shape_body_torch.shape[0])
        body_ids[valid_shapes] = self.shape_body_torch[shapes[valid_shapes].long()]
        return body_ids

    def _is_primary_body(self, body_ids: torch.Tensor) -> torch.Tensor:
        """Return whether each body belongs to the primary articulation."""
        result = torch.zeros_like(body_ids, dtype=torch.bool)
        valid = (body_ids >= 0) & (body_ids < self.primary_body_mask.shape[0])
        result[valid] = self.primary_body_mask[body_ids[valid]]
        return result

    def _derive_primary_body_mask(self) -> tuple[torch.Tensor, str]:
        """Identify the first articulation in every world as the primary robot."""
        articulation_array = self.model.joint_articulation
        parent_array = self.model.joint_parent
        child_array = self.model.joint_child
        world_array = self.model.joint_world
        if any(array is None for array in (articulation_array, parent_array, child_array, world_array)):
            return torch.ones(int(self.model.body_count), dtype=torch.bool, device=self.device), "dynamic_body_first"

        assert articulation_array is not None
        assert parent_array is not None
        assert child_array is not None
        assert world_array is not None
        joint_articulation = articulation_array.numpy()
        joint_parent = parent_array.numpy()
        joint_child = child_array.numpy()
        joint_world = world_array.numpy()
        primary_bodies = torch.zeros(int(self.model.body_count), dtype=torch.bool, device=self.device)
        for world_id in range(self.num_envs):
            world_joints = (joint_world == world_id) & (joint_articulation >= 0)
            articulation_ids = joint_articulation[world_joints]
            if articulation_ids.size == 0:
                continue
            primary_articulation = articulation_ids.min()
            primary_joints = world_joints & (joint_articulation == primary_articulation)
            for body_id in (*joint_parent[primary_joints].tolist(), *joint_child[primary_joints].tolist()):
                if body_id >= 0:
                    primary_bodies[int(body_id)] = True
        return primary_bodies, "primary_articulation_first"

    def _read_thickness(
        self,
        contacts: newton.Contacts,
        contact_count: int,
        side: Literal[0, 1],
    ) -> torch.Tensor:
        margin_name = f"rigid_contact_margin{side}"
        margins = getattr(contacts, margin_name)
        if margins is None:
            raise ValueError(
                f"NewtonContactAdapter requires {margin_name} for thickness. "
                "Please pass contacts generated by Newton's collision pipeline."
            )

        return wp.to_torch(margins)[:contact_count].to(self.device)

    def _read_optional_vec3(self, contacts: newton.Contacts, name: str, contact_count: int) -> torch.Tensor | None:
        """Return an optional Newton contact vec3 buffer, or ``None`` if absent."""
        values = getattr(contacts, name, None)
        if values is None:
            return None
        return wp.to_torch(values)[:contact_count].to(self.device)

    def _read_separation_and_world_points(
        self,
        shape0: torch.Tensor,
        shape1: torch.Tensor,
        point0: torch.Tensor,
        point1: torch.Tensor,
        normal: torch.Tensor,
        thickness0: torch.Tensor,
        thickness1: torch.Tensor,
        offset0: torch.Tensor | None,
        offset1: torch.Tensor | None,
        state: newton.State | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state is None:
            raise ValueError(
                "NewtonContactAdapter computes surface separation from geometry. "
                "Pass the current Newton State to NewtonContactAdapter.update() "
                "so world-space separation can be computed from point0/point1/normal."
            )

        point0_world = self._points_to_world(shape0, point0, state)
        point1_world = self._points_to_world(shape1, point1, state)
        surface0_local = point0 if offset0 is None else point0 + offset0
        surface1_local = point1 if offset1 is None else point1 + offset1
        surface0_world = self._points_to_world(shape0, surface0_local, state)
        surface1_world = self._points_to_world(shape1, surface1_local, state)
        normal_distance = torch.sum(normal * (surface1_world - surface0_world), dim=-1)
        surface_separation = normal_distance - thickness0 - thickness1
        return surface_separation, point0_world, point1_world, surface0_world, surface1_world

    def _points_to_world(
        self,
        shapes: torch.Tensor,
        points: torch.Tensor,
        state: newton.State,
    ) -> torch.Tensor:
        if state.body_q is None:
            raise ValueError("NewtonContactAdapter requires state.body_q.")
        body_q = wp.to_torch(state.body_q).to(self.device)
        points_world = points.clone()

        body_ids = self._shape_body_ids(shapes)
        dynamic_mask = body_ids >= 0
        if dynamic_mask.any():
            q = body_q[body_ids[dynamic_mask]]
            points_world[dynamic_mask] = torch_utils.transform_point(
                q[:, :3],
                q[:, 3:7],
                points[dynamic_mask],
            )

        return points_world

    def _contact_env_id(self, shape0: int, shape1: int) -> int:
        env_id = self._shape_to_env_id(shape0)
        if env_id >= 0:
            return env_id
        return self._shape_to_env_id(shape1)

    def _contact_env_ids(self, shape0: torch.Tensor, shape1: torch.Tensor) -> torch.Tensor:
        """Vectorized env id for each contact (``-1`` when unresolved)."""
        env0 = self._shapes_to_env_ids(shape0)
        env1 = self._shapes_to_env_ids(shape1)
        return torch.where(env0 >= 0, env0, env1)

    def _shapes_to_env_ids(self, shapes: torch.Tensor) -> torch.Tensor:
        body_ids = self._shape_body_ids(shapes)
        env_ids = torch.full(body_ids.shape, -1, dtype=torch.long, device=shapes.device)
        valid = body_ids >= 0
        if not bool(valid.any()):
            return env_ids
        if getattr(self, "body_world_torch", None) is not None:
            env_ids[valid] = self.body_world_torch[body_ids[valid]]
        elif self.body_world is not None:
            body_world = torch.as_tensor(self.body_world, dtype=torch.long, device=shapes.device)
            env_ids[valid] = body_world[body_ids[valid]]
        else:
            env_ids[valid] = body_ids[valid] // self.bodies_per_env
        return env_ids

    def _shape_to_env_id(self, shape_id: int) -> int:
        if shape_id < 0 or shape_id >= len(self.shape_body):
            return -1

        body_id = int(self.shape_body[shape_id])
        if body_id < 0:
            return -1

        if self.body_world is not None:
            return int(self.body_world[body_id])

        return body_id // self.bodies_per_env
