# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
from typing import Literal

import newton
import numpy as np
import torch
import warp as wp

from isaaclab_neural.contacts.contact_set_encoder import ContactSetEncoder
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_FILTER_NONE,
    CONTACT_FILTER_SOLVER_ACTIVE,
    CONTACT_FILTER_SOLVER_ACTIVE_MOVABLE,
    CONTACT_REPRESENTATION_FLAT,
    CONTACT_REPRESENTATION_TOKENS,
    CONTACT_TOKEN_DIM,
    DEFAULT_MAX_CONTACT_TOKENS,
    NATIVE_CONTACT_FLOAT_FIELDS,
    NATIVE_CONTACT_INTEGER_FIELDS,
    NATIVE_CONTACT_SCALAR_FIELDS,
    NATIVE_CONTACT_VECTOR_FIELDS,
)
from isaaclab_neural.contacts.kernels import transform_native_contact_geometry
from isaaclab_neural.contacts.native_pair_projector import (
    gather_or_fill,
    native_pair_filter_mask,
    project_native_pairs,
)
from isaaclab_neural.contacts.packing import ContactPackingPolicy
from isaaclab_neural.utils import step_profile, torch_utils

logger = logging.getLogger(__name__)


def find_immovable_primary_slots(model: newton.Model, primary_ids: torch.Tensor) -> tuple[int, ...]:
    """Return immovable primary slots; fixed joints inherit their parent's mobility."""
    required = (model.joint_parent, model.joint_child, model.joint_qd_start, model.body_flags)
    if any(value is None for value in required):
        raise ValueError("Contact mobility classification requires the joint tree and body flags.")
    parents, children, starts, flags = (value.numpy() for value in required if value is not None)
    joints = {int(child): index for index, child in enumerate(children) if child >= 0}
    if len(joints) != int((children >= 0).sum()):
        raise ValueError("Contact mobility requires an unambiguous body joint tree.")
    movable = np.full(model.body_count, -1, dtype=np.int8)

    def resolve(body: int) -> bool:
        if body < 0:
            return False
        if movable[body] == -2:
            raise ValueError("Cyclic body ancestry in contact mobility classification.")
        if movable[body] >= 0:
            return bool(movable[body])
        movable[body] = -2
        joint = joints.get(body)
        value = not bool(int(flags[body]) & int(newton.BodyFlags.KINEMATIC)) and (
            joint is None or starts[joint + 1] > starts[joint] or resolve(int(parents[joint]))
        )
        movable[body] = int(value)
        return value

    for body in range(model.body_count):
        resolve(body)
    fixed = movable[primary_ids.detach().cpu().numpy()] == 0
    if fixed.ndim != 2 or not np.all(fixed == fixed[:1]):
        raise ValueError("Contact fixed-body slots must be uniform across worlds.")
    return tuple(int(slot) for slot in np.flatnonzero(fixed[0]))


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
        max_contact_tokens: int = DEFAULT_MAX_CONTACT_TOKENS,
        contact_filter: str = CONTACT_FILTER_NONE,
        retain_native_pairs: bool = True,
    ):
        """Args (beyond the layout settings):
        retain_native_pairs: Keep every resolved native pair for :meth:`native_contact_pairs` (data
            collection). When False and a contact filter is set, pairs the filter would reject are dropped
            before per-pair geometry is built; packed outputs are identical, but :meth:`native_contact_pairs`
            is unavailable. Used by the online NeRD solver.
        """
        self.model = model
        self.retain_native_pairs = bool(retain_native_pairs)
        self.num_envs = int(model.world_count)
        self.contact_representation = contact_representation
        self.max_contact_tokens = int(max_contact_tokens)
        if self.contact_representation == CONTACT_REPRESENTATION_TOKENS:
            self.num_contacts_per_env = self.max_contact_tokens
        else:
            self.num_contacts_per_env = int(num_contacts_per_env)
        self.num_total_contacts = self.num_envs * self.num_contacts_per_env
        self.packing_policy: ContactPackingPolicy = packing_policy
        if contact_filter not in {
            CONTACT_FILTER_NONE,
            CONTACT_FILTER_SOLVER_ACTIVE,
            CONTACT_FILTER_SOLVER_ACTIVE_MOVABLE,
        }:
            raise ValueError(f"Unsupported contact_filter: {contact_filter!r}.")
        self.contact_filter = contact_filter
        if (
            contact_filter == CONTACT_FILTER_SOLVER_ACTIVE_MOVABLE
            and contact_representation != CONTACT_REPRESENTATION_TOKENS
        ):
            raise ValueError("Movable filtering requires contact-token representation.")

        if device is None:
            self.device = torch.device(wp.device_to_torch(model.device))
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
        self.primary_body_ids_by_world = torch.stack(
            [
                torch.nonzero(
                    self.primary_body_mask
                    & (
                        self.body_world_torch == world_id
                        if self.body_world_torch is not None
                        else (
                            torch.arange(int(model.body_count), device=self.device) // self.bodies_per_env == world_id
                        )
                    ),
                    as_tuple=False,
                ).reshape(-1)
                for world_id in range(self.num_envs)
            ]
        )
        self.fixed_body_slots = (
            find_immovable_primary_slots(model, self.primary_body_ids_by_world)
            if contact_filter == CONTACT_FILTER_SOLVER_ACTIVE_MOVABLE
            else None
        )
        body_local_ids = np.empty(int(model.body_count), dtype=np.int64)
        if self.body_world is None:
            body_local_ids[:] = np.arange(int(model.body_count)) % self.bodies_per_env
        else:
            for world_id in range(self.num_envs):
                world_bodies = np.flatnonzero(self.body_world == world_id)
                body_local_ids[world_bodies] = np.arange(world_bodies.size)
        self.body_local_ids = torch.as_tensor(body_local_ids, dtype=torch.long, device=self.device)
        self.primary_body_q = torch.zeros(
            (self.num_envs, self.num_primary_bodies_per_env, 7),
            dtype=torch.float32,
            device=self.device,
        )
        self.primary_body_q[..., 6] = 1.0
        self._contact_frames = 0
        self._raw_contacts_total = 0
        self._raw_contacts_gpu = torch.zeros((), dtype=torch.long, device=self.device)
        self._packed_contacts_total = 0
        self._dropped_contacts_total = 0
        self._truncated_frames = 0
        self._truncation_warning_emitted = False
        # Deferred GPU counters (synced only in truncation_summary / warnings).
        self._packed_contacts_gpu = torch.zeros((), dtype=torch.long, device=self.device)
        self._filtered_contacts_gpu = torch.zeros((), dtype=torch.long, device=self.device)
        self._dropped_contacts_gpu = torch.zeros((), dtype=torch.long, device=self.device)
        self._truncated_frames_gpu = torch.zeros((), dtype=torch.long, device=self.device)
        self._max_required_tokens_gpu = torch.zeros((), dtype=torch.long, device=self.device)
        self._torch_view_cache: dict[int, tuple[object, torch.Tensor]] = {}
        self.use_warp_geometry = self.device.type == "cuda" and os.environ.get("NERD_WARP_CONTACT_GEOMETRY", "1") != "0"
        self._geometry_capacity = 0
        self._geometry_buffers: dict[str, torch.Tensor] = {}
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
        self._native_pair_encoder = ContactSetEncoder(
            model=model,
            primary_body_mask=self.primary_body_mask,
            shape_body_torch=self.shape_body_torch,
            bodies_per_env=self.bodies_per_env,
            num_envs=self.num_envs,
            max_contact_tokens=self.max_contact_tokens,
            device=self.device,
            primary_body_ids_by_world=self.primary_body_ids_by_world,
        )
        self._native_contact_pairs = self._empty_native_contact_pairs()

    def reset_buffers(self) -> None:
        """Clear all fixed-size contact buffers before packing a new frame."""
        if self.contact_representation == CONTACT_REPRESENTATION_TOKENS:
            self.contact_tokens.zero_()
            self.contact_token_overflow.zero_()
            if hasattr(self, "contact_token_body_ids"):
                self.contact_token_body_ids.fill_(-1)
            return

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

    def update(
        self,
        contacts: newton.Contacts,
        state: newton.State | None = None,
    ) -> None:
        """Pack Newton native contacts into fixed NeRD contact buffers."""
        if hasattr(self, "primary_body_q"):
            self._update_primary_body_q(state)
        contact_count = self._contact_count(contacts)
        self._contact_frames += 1
        if contact_count == 0:
            self._native_contact_pairs = self._empty_native_contact_pairs()
            self.reset_buffers()
            return
        if self.contact_representation != CONTACT_REPRESENTATION_TOKENS:
            self.reset_buffers()

        with step_profile.section("adapter_read_raw"):
            raw = self._read_raw_contacts(contacts, contact_count, state)
        if (
            not getattr(self, "retain_native_pairs", True)
            and self.contact_filter != CONTACT_FILTER_NONE
            and state is not None
        ):
            # Drop pairs the input filter rejects before building their geometry. Pair order is preserved,
            # so packing (which applies the same filter) produces identical outputs.
            resolved, eligible = self._prefilter_masks(raw)
            kept = torch.nonzero(eligible, as_tuple=False).reshape(-1)
            count = raw["shape0"].shape[0]
            raw = {
                name: value.index_select(0, kept) if value.shape[:1] == (count,) and not name.startswith("_") else value
                for name, value in raw.items()
            }
            self._native_contact_pairs = self._build_native_contact_pairs(raw, state)
            frame_raw = 0
            frame_raw_tensor = resolved.sum()
            self._raw_contacts_gpu += frame_raw_tensor
            frame_eligible = eligible.sum()
        else:
            self._native_contact_pairs = self._build_native_contact_pairs(raw, state)
            frame_raw = self._native_contact_pairs["world_ids"].shape[0]
            frame_raw_tensor = torch.as_tensor(frame_raw, device=self.device, dtype=torch.long)
            if self.contact_filter != CONTACT_FILTER_NONE:
                frame_eligible = native_pair_filter_mask(
                    self._native_contact_pairs, self.contact_filter, self.fixed_body_slots
                ).sum()
            else:
                frame_eligible = frame_raw_tensor
        self._filtered_contacts_gpu += frame_raw_tensor - frame_eligible
        if self.contact_representation == CONTACT_REPRESENTATION_TOKENS:
            if state is None:
                raise ValueError("Contact token encoding requires the current Newton State.")
            with step_profile.section("adapter_encode_tokens"):
                projected = project_native_pairs(
                    self._native_contact_pairs,
                    num_frames=self.num_envs,
                    representation=self.contact_representation,
                    packing_policy="body_round_robin_pair_atomic",
                    max_contact_tokens=self.max_contact_tokens,
                    contact_filter=self.contact_filter,
                    fixed_body_slots=self.fixed_body_slots,
                )
                self.contact_tokens.copy_(projected["contact_tokens"])
                self.contact_token_overflow.copy_(projected["contact_token_overflow"])
                self.contact_token_body_ids.copy_(projected["contact_token_body_ids"])
            packed_per_env = (self.contact_tokens[..., 0] > 0.5).sum(dim=-1)
            valid_tokens = packed_per_env.sum()
            self._max_required_tokens_gpu.copy_(
                torch.maximum(self._max_required_tokens_gpu, (packed_per_env + self.contact_token_overflow).max())
            )
            overflow_sum = self.contact_token_overflow.sum()
            self._raw_contacts_total += frame_raw
            self._packed_contacts_gpu += valid_tokens.to(dtype=torch.long)
            dropped = overflow_sum.to(dtype=torch.long)
            self._dropped_contacts_gpu += dropped
            self._truncated_frames_gpu += (dropped > 0).to(dtype=torch.long)
            if self.device.type == "cpu" and not self._truncation_warning_emitted and int(dropped.item()) > 0:
                logger.warning(
                    "Contact token packing dropped tokens because max_contact_tokens=%d. "
                    "Further truncation warnings are suppressed; inspect truncation_summary() for totals.",
                    self.max_contact_tokens,
                )
                self._truncation_warning_emitted = True
            return

        with step_profile.section("adapter_pack_flat"):
            projected = project_native_pairs(
                self._native_contact_pairs,
                num_frames=self.num_envs,
                representation=self.contact_representation,
                packing_policy=self.packing_policy,
                flat_capacity=self.num_contacts_per_env,
                return_pair_indices=True,
                contact_filter=self.contact_filter,
            )
            self.contact_masks.copy_(projected["contact_masks"])
            self.contact_normals.copy_(
                projected["contact_normals"].reshape(self.num_envs, self.num_contacts_per_env, 3)
            )
            self.contact_depths.copy_(projected["contact_depths"])
            self.contact_thicknesses_0.copy_(projected["contact_thicknesses_0"])
            self.contact_thicknesses_1.copy_(projected["contact_thicknesses_1"])
            self.contact_points_0.copy_(
                projected["contact_points_0"].reshape(self.num_envs, self.num_contacts_per_env, 3)
            )
            self.contact_points_1.copy_(
                projected["contact_points_1"].reshape(self.num_envs, self.num_contacts_per_env, 3)
            )
            pair_indices = projected["_pair_indices"]
            selected = pair_indices >= 0
            if hasattr(self, "contact_body_ids_0") and hasattr(self, "body_local_ids") and selected.any():
                source = pair_indices[selected]
                body0 = self._native_contact_pairs["body_ids_0"][source]
                body1 = self._native_contact_pairs["body_ids_1"][source]
                valid_body0 = body0 >= 0
                valid_body1 = body1 >= 0
                selected0 = selected.clone()
                selected1 = selected.clone()
                selected0[selected] = valid_body0
                selected1[selected] = valid_body1
                self.contact_body_ids_0[selected0] = self.body_local_ids[body0[valid_body0]]
                self.contact_body_ids_1[selected1] = self.body_local_ids[body1[valid_body1]]
        packed = self.contact_masks.sum()
        dropped = frame_eligible - packed
        self._raw_contacts_total += frame_raw
        self._packed_contacts_gpu += packed.to(dtype=torch.long)
        self._dropped_contacts_gpu += dropped
        self._truncated_frames_gpu += (dropped > 0).to(dtype=torch.long)
        if self.device.type == "cpu" and not self._truncation_warning_emitted and int(dropped.item()) > 0:
            logger.warning(
                "Newton-native contact packing dropped %d contacts because num_contacts_per_env=%d. "
                "Further truncation warnings are suppressed; inspect truncation_summary() for totals.",
                int(dropped.item()),
                self.num_contacts_per_env,
            )
            self._truncation_warning_emitted = True

    def native_contact_pairs(self) -> dict[str, torch.Tensor]:
        """Return the current canonical native contact-pair tensors.

        The returned tensors contain only pairs whose Newton world can be
        resolved. They are collected before fixed-capacity contact packing.
        Callers retaining a frame must clone the tensors before the next
        :meth:`update`.
        """
        if not getattr(self, "retain_native_pairs", True):
            raise RuntimeError("This adapter drops filtered native pairs (retain_native_pairs=False).")
        return self._native_contact_pairs

    def _prefilter_masks(self, raw: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (resolved, filter-eligible) masks over raw pairs, matching :meth:`_build_native_contact_pairs`."""
        encoder = self._native_pair_encoder
        shape0, shape1 = raw["shape0"], raw["shape1"]
        body0, body1 = encoder._shape_body_ids(shape0), encoder._shape_body_ids(shape1)
        dynamic0, dynamic1 = body0 >= 0, body1 >= 0
        world0, world1 = encoder._contact_env_ids(body0, shape0), encoder._contact_env_ids(body1, shape1)
        world = torch.where(dynamic0, world0, torch.where(dynamic1, world1, torch.full_like(world0, -1)))
        normal_norm = torch.linalg.vector_norm(raw["normal"], dim=-1)
        resolved = (
            (world >= 0)
            & (world < self.num_envs)
            & torch.isfinite(normal_norm)
            & (normal_norm > torch.finfo(raw["normal"].dtype).eps)
        )
        pairs = {
            "separation": raw["surface_separation"],
            "body_ids_0": body0,
            "body_ids_1": body1,
            "body_slots_0": encoder.body_slot(body0),
            "body_slots_1": encoder.body_slot(body1),
        }
        return resolved, resolved & native_pair_filter_mask(pairs, self.contact_filter, self.fixed_body_slots)

    def _empty_native_contact_pairs(self) -> dict[str, torch.Tensor]:
        """Return a schema-complete empty native contact-pair mapping."""
        pairs: dict[str, torch.Tensor] = {
            name: torch.empty(0, dtype=torch.int64, device=self.device) for name in NATIVE_CONTACT_INTEGER_FIELDS
        }
        pairs.update(
            {
                name: torch.empty((0, 3), dtype=torch.float32, device=self.device)
                for name in NATIVE_CONTACT_VECTOR_FIELDS
            }
        )
        pairs.update(
            {name: torch.empty(0, dtype=torch.float32, device=self.device) for name in NATIVE_CONTACT_SCALAR_FIELDS}
        )
        return pairs

    def _build_native_contact_pairs(
        self,
        raw: dict[str, torch.Tensor],
        state: newton.State | None,
    ) -> dict[str, torch.Tensor]:
        """Build the unresolved-capacity canonical native pair view."""
        pair_count = raw["shape0"].shape[0]
        body_ids_0 = self._shape_body_ids(raw["shape0"])
        body_ids_1 = self._shape_body_ids(raw["shape1"])
        world_ids = self._contact_env_ids(raw["shape0"], raw["shape1"])
        if state is not None:
            rows = self._native_pair_encoder.directed_rows(raw, state)
            first = slice(0, pair_count)
            second = slice(pair_count, 2 * pair_count)
            body_ids_0 = rows.body_id[first]
            body_ids_1 = rows.body_id[second]
            world_ids = rows.world_id[first]
            body_slots_0 = self._native_pair_encoder.body_slot(body_ids_0)
            body_slots_1 = self._native_pair_encoder.body_slot(body_ids_1)
            lever_arms_0_w = rows.lever_world[first]
            lever_arms_1_w = rows.lever_world[second]
            relative_velocities_0_w = rows.velocity_world[first]
            relative_velocities_1_w = rows.velocity_world[second]
        else:
            body_slots_0 = torch.full_like(body_ids_0, -1)
            body_slots_1 = torch.full_like(body_ids_1, -1)
            vector_zeros = torch.zeros((pair_count, 3), dtype=torch.float32, device=self.device)
            lever_arms_0_w = vector_zeros
            lever_arms_1_w = vector_zeros
            relative_velocities_0_w = vector_zeros
            relative_velocities_1_w = vector_zeros
        normal = raw["normal"]
        normal_norm = torch.linalg.vector_norm(normal, dim=-1)
        valid = (
            (world_ids >= 0)
            & (world_ids < self.num_envs)
            & torch.isfinite(normal_norm)
            & (normal_norm > torch.finfo(normal.dtype).eps)
        )

        point0_world = raw["point0_world"]
        point1_world = raw["point1_world"]
        point0_local = raw.get("point0", point0_world)
        point1_local = raw.get("point1", point1_world)
        surface0_world = raw.get("surface0_world", point0_world)
        surface1_world = raw.get("surface1_world", point1_world)
        offset0 = raw.get("offset0", torch.zeros_like(point0_local))
        offset1 = raw.get("offset1", torch.zeros_like(point1_local))
        # One nonzero (one host sync) instead of a synchronizing boolean mask per field; order is unchanged.
        kept = torch.nonzero(valid, as_tuple=False).reshape(-1)
        pairs = {
            "shape_ids_0": raw["shape0"].index_select(0, kept).to(dtype=torch.int64),
            "shape_ids_1": raw["shape1"].index_select(0, kept).to(dtype=torch.int64),
            "body_ids_0": body_ids_0.index_select(0, kept).to(dtype=torch.int64),
            "body_ids_1": body_ids_1.index_select(0, kept).to(dtype=torch.int64),
            "body_slots_0": body_slots_0.index_select(0, kept).to(dtype=torch.int64),
            "body_slots_1": body_slots_1.index_select(0, kept).to(dtype=torch.int64),
            "world_ids": world_ids.index_select(0, kept).to(dtype=torch.int64),
            "points_0_local": point0_local.index_select(0, kept).to(dtype=torch.float32),
            "points_1_local": point1_local.index_select(0, kept).to(dtype=torch.float32),
            "points_0_w": point0_world.index_select(0, kept).to(dtype=torch.float32),
            "points_1_w": point1_world.index_select(0, kept).to(dtype=torch.float32),
            "surface_points_0_w": surface0_world.index_select(0, kept).to(dtype=torch.float32),
            "surface_points_1_w": surface1_world.index_select(0, kept).to(dtype=torch.float32),
            "normal_w": normal.index_select(0, kept).to(dtype=torch.float32),
            "separation": raw["surface_separation"].index_select(0, kept).to(dtype=torch.float32),
            "thickness_0": raw["thickness0"].index_select(0, kept).to(dtype=torch.float32),
            "thickness_1": raw["thickness1"].index_select(0, kept).to(dtype=torch.float32),
            "offsets_0": offset0.index_select(0, kept).to(dtype=torch.float32),
            "offsets_1": offset1.index_select(0, kept).to(dtype=torch.float32),
            "lever_arms_0_w": lever_arms_0_w.index_select(0, kept).to(dtype=torch.float32),
            "lever_arms_1_w": lever_arms_1_w.index_select(0, kept).to(dtype=torch.float32),
            "relative_velocities_0_w": relative_velocities_0_w.index_select(0, kept).to(dtype=torch.float32),
            "relative_velocities_1_w": relative_velocities_1_w.index_select(0, kept).to(dtype=torch.float32),
        }
        if set(pairs) != set(NATIVE_CONTACT_INTEGER_FIELDS + NATIVE_CONTACT_FLOAT_FIELDS):
            raise RuntimeError("Native contact-pair fields do not match the schema.")
        return pairs

    def _update_primary_body_q(self, state: newton.State | None) -> None:
        """Copy current primary-body poses from the Newton state."""
        if state is None or getattr(state, "body_q", None) is None:
            self.primary_body_q.zero_()
            self.primary_body_q[..., 6] = 1.0
            return
        body_q = self._warp_torch_view(state.body_q)
        ids = self.primary_body_ids_by_world.reshape(-1)
        gathered = body_q.index_select(0, ids).reshape(self.num_envs, self.num_primary_bodies_per_env, 7)
        self.primary_body_q.copy_(gathered)

    def to_neural_inputs(self) -> dict[str, torch.Tensor]:
        """Return contact tensors shaped like fixed-ground neural inputs.

        Returns views into the adapter buffers. Callers that retain contacts
        across frames (e.g. transformer history) must clone before the next
        :meth:`update`, which :class:`TransformerNeuralSolver` already does.
        """
        if self.contact_representation == CONTACT_REPRESENTATION_TOKENS:
            return {
                "contact_tokens": self.contact_tokens,
                "contact_token_overflow": self.contact_token_overflow,
            }

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
        filtered_gpu = int(getattr(self, "_filtered_contacts_gpu", torch.tensor(0)).item())
        dropped_gpu = int(getattr(self, "_dropped_contacts_gpu", torch.tensor(0)).item())
        truncated_gpu = int(getattr(self, "_truncated_frames_gpu", torch.tensor(0)).item())
        packed = self._packed_contacts_total + packed_gpu
        dropped = self._dropped_contacts_total + dropped_gpu
        truncated = self._truncated_frames + truncated_gpu
        if dropped > 0 and not self._truncation_warning_emitted:
            logger.warning(
                "Native contact packing dropped inputs because the configured per-environment capacity was exceeded. "
                "Further truncation warnings are suppressed; inspect truncation_summary() for totals."
            )
            self._truncation_warning_emitted = True
        summary = {
            "frames": self._contact_frames,
            "raw_contacts": self._raw_contacts_total + int(getattr(self, "_raw_contacts_gpu", torch.tensor(0)).item()),
            "packed_contacts": packed,
            "dropped_contacts": dropped,
            "truncated_frames": truncated,
            "truncated_frame_ratio": truncated / max(self._contact_frames, 1),
        }
        if self.contact_filter != CONTACT_FILTER_NONE:
            summary["filtered_contacts"] = filtered_gpu
        if self.contact_representation == CONTACT_REPRESENTATION_TOKENS:
            summary.update(
                {
                    "overflow_tokens_total": dropped,
                    "max_tokens_required_per_env": int(self._max_required_tokens_gpu.item()),
                    "overflow_frames": int((self.contact_token_overflow > 0).sum().item()),
                }
            )
        return summary

    def _contact_count(self, contacts: newton.Contacts) -> int:
        count_array = contacts.rigid_contact_count
        if count_array is None:
            return 0
        count = int(self._warp_torch_view(count_array)[0].item())
        return min(count, int(contacts.rigid_contact_max))

    def _read_raw_contacts(
        self,
        contacts: newton.Contacts,
        contact_count: int,
        state: newton.State | None,
    ) -> dict[str, torch.Tensor]:
        shape0 = self._warp_torch_view(contacts.rigid_contact_shape0)[:contact_count]
        shape1 = self._warp_torch_view(contacts.rigid_contact_shape1)[:contact_count]
        point0 = self._warp_torch_view(contacts.rigid_contact_point0)[:contact_count]
        point1 = self._warp_torch_view(contacts.rigid_contact_point1)[:contact_count]
        normal = self._warp_torch_view(contacts.rigid_contact_normal)[:contact_count]

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
        if offset0 is None:
            offset0 = torch.zeros_like(point0)
            offset1 = torch.zeros_like(point1)

        raw = {
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
            "offset0": offset0,
            "offset1": offset1,
        }
        if state is not None:
            if state.body_q is not None:
                raw["_body_q"] = self._warp_torch_view(state.body_q)
            if state.body_qd is not None:
                raw["_body_qd"] = self._warp_torch_view(state.body_qd)
        return raw

    def canonicalize_contact_sides(
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
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
        if offset0 is None or offset1 is None:
            canonical_offset0 = None
            canonical_offset1 = None
        else:
            canonical_offset0 = torch.where(vector_swap, offset1, offset0)
            canonical_offset1 = torch.where(vector_swap, offset0, offset1)
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

    _canonicalize_contact_sides = canonicalize_contact_sides

    def _shape_body_ids(self, shapes: torch.Tensor) -> torch.Tensor:
        """Return owning body indices, using ``-1`` for static or invalid shapes."""
        return gather_or_fill(self.shape_body_torch, shapes, -1)

    def _is_primary_body(self, body_ids: torch.Tensor) -> torch.Tensor:
        """Return whether each body is an articulated contact owner."""
        return gather_or_fill(self.primary_body_mask, body_ids, False)

    def _derive_primary_body_mask(self) -> tuple[torch.Tensor, str]:
        """Mark every articulated body in each world as a contact owner."""
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
            if not np.any(world_joints):
                continue
            for body_id in (*joint_parent[world_joints].tolist(), *joint_child[world_joints].tolist()):
                if body_id >= 0:
                    primary_bodies[int(body_id)] = True
        return primary_bodies, "all_articulation_bodies"

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

        return self._warp_torch_view(margins)[:contact_count]

    def _read_optional_vec3(self, contacts: newton.Contacts, name: str, contact_count: int) -> torch.Tensor | None:
        """Return an optional Newton contact vec3 buffer, or ``None`` if absent."""
        values = getattr(contacts, name, None)
        if values is None:
            return None
        return self._warp_torch_view(values)[:contact_count]

    def _warp_torch_view(self, array) -> torch.Tensor:
        """Return a cached Torch view of a Warp array on the adapter device."""
        key = id(array)
        cached = self._torch_view_cache.get(key)
        if cached is None or cached[0] is not array:
            tensor = wp.to_torch(array)
            if tensor.device != self.device:
                tensor = tensor.to(self.device)
            self._torch_view_cache[key] = (array, tensor)
            return tensor
        return cached[1]

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
        if getattr(self, "use_warp_geometry", False):
            return self._read_separation_and_world_points_warp(
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

        point0_world = self._points_to_world(shape0, point0, state)
        point1_world = self._points_to_world(shape1, point1, state)
        surface0_local = point0 if offset0 is None else point0 + offset0
        surface1_local = point1 if offset1 is None else point1 + offset1
        surface0_world = self._points_to_world(shape0, surface0_local, state)
        surface1_world = self._points_to_world(shape1, surface1_local, state)
        normal_distance = torch.sum(normal * (point1_world - point0_world), dim=-1)
        surface_separation = normal_distance - thickness0 - thickness1
        return surface_separation, point0_world, point1_world, surface0_world, surface1_world

    def _read_separation_and_world_points_warp(
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
        state: newton.State,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute world-space contact geometry with one fused Warp launch."""
        if state.body_q is None:
            raise ValueError("NewtonContactAdapter requires state.body_q.")
        count = shape0.shape[0]
        buffers = self._geometry_scratch(count)
        dummy_offset0 = point0 if offset0 is None else offset0
        dummy_offset1 = point1 if offset1 is None else offset1
        wp.launch(
            transform_native_contact_geometry,
            dim=count,
            inputs=[
                wp.from_torch(shape0),
                wp.from_torch(shape1),
                wp.from_torch(point0, dtype=wp.vec3),
                wp.from_torch(point1, dtype=wp.vec3),
                wp.from_torch(normal, dtype=wp.vec3),
                wp.from_torch(thickness0),
                wp.from_torch(thickness1),
                wp.from_torch(dummy_offset0, dtype=wp.vec3),
                wp.from_torch(dummy_offset1, dtype=wp.vec3),
                offset0 is not None,
                wp.from_torch(self.shape_body_torch),
                state.body_q,
            ],
            outputs=[
                wp.from_torch(buffers["point0_world"], dtype=wp.vec3),
                wp.from_torch(buffers["point1_world"], dtype=wp.vec3),
                wp.from_torch(buffers["surface0_world"], dtype=wp.vec3),
                wp.from_torch(buffers["surface1_world"], dtype=wp.vec3),
                wp.from_torch(buffers["surface_separation"]),
            ],
            device=self.model.device,
        )
        return (
            buffers["surface_separation"],
            buffers["point0_world"],
            buffers["point1_world"],
            buffers["surface0_world"],
            buffers["surface1_world"],
        )

    def _geometry_scratch(self, count: int) -> dict[str, torch.Tensor]:
        """Return reusable world-space geometry buffers sized for ``count`` contacts."""
        if count > self._geometry_capacity:
            self._geometry_capacity = max(count, max(16, 2 * self._geometry_capacity))
            with torch.inference_mode(False):
                self._geometry_buffers = {
                    "point0_world": torch.empty((self._geometry_capacity, 3), device=self.device),
                    "point1_world": torch.empty((self._geometry_capacity, 3), device=self.device),
                    "surface0_world": torch.empty((self._geometry_capacity, 3), device=self.device),
                    "surface1_world": torch.empty((self._geometry_capacity, 3), device=self.device),
                    "surface_separation": torch.empty(self._geometry_capacity, device=self.device),
                }
        return {name: value[:count] for name, value in self._geometry_buffers.items()}

    def _points_to_world(
        self,
        shapes: torch.Tensor,
        points: torch.Tensor,
        state: newton.State,
    ) -> torch.Tensor:
        if state.body_q is None:
            raise ValueError("NewtonContactAdapter requires state.body_q.")
        body_q = self._warp_torch_view(state.body_q)

        body_ids = self._shape_body_ids(shapes)
        dynamic_mask = body_ids >= 0
        safe_body_ids = body_ids.clamp(min=0, max=body_q.shape[0] - 1)
        q = body_q.index_select(0, safe_body_ids)
        transformed = torch_utils.transform_point(q[:, :3], q[:, 3:7], points)
        return torch.where(dynamic_mask.unsqueeze(-1), transformed, points)

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
        body_world_torch = getattr(self, "body_world_torch", None)
        if body_world_torch is None and self.body_world is not None:
            body_world_torch = torch.as_tensor(self.body_world, dtype=torch.long, device=shapes.device)
        if body_world_torch is not None:
            return gather_or_fill(body_world_torch, body_ids, -1)
        return torch.where(body_ids >= 0, body_ids // self.bodies_per_env, torch.full_like(body_ids, -1))

    def _shape_to_env_id(self, shape_id: int) -> int:
        if shape_id < 0 or shape_id >= len(self.shape_body):
            return -1

        body_id = int(self.shape_body[shape_id])
        if body_id < 0:
            return -1

        if self.body_world is not None:
            return int(self.body_world[body_id])

        return body_id // self.bodies_per_env
