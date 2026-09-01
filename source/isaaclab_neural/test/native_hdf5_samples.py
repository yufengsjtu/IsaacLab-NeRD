# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared native-contact rollout samples for HDF5 tests."""

from __future__ import annotations

import torch
from isaaclab_neural.contacts.contact_set_schema import (
    NATIVE_CONTACT_INTEGER_FIELDS,
    NATIVE_CONTACT_SCALAR_FIELDS,
    NATIVE_CONTACT_VECTOR_FIELDS,
)


def make_native_rollouts(
    *,
    num_trajectories: int = 2,
    num_steps: int = 3,
    trajectory_start: int = 0,
) -> dict:
    """Create complete native-pair CSR rollouts with one pair per frame."""
    trajectory_ids = torch.arange(
        trajectory_start,
        trajectory_start + num_trajectories,
        dtype=torch.int64,
    )
    frame_ids = (trajectory_ids[:, None] * num_steps + torch.arange(num_steps, dtype=torch.int64)[None, :]).reshape(-1)
    pair_count = frame_ids.numel()

    native_contacts = {name: torch.zeros(pair_count, dtype=torch.int64) for name in NATIVE_CONTACT_INTEGER_FIELDS}
    native_contacts.update(
        {name: torch.zeros((pair_count, 3), dtype=torch.float32) for name in NATIVE_CONTACT_VECTOR_FIELDS}
    )
    native_contacts.update(
        {name: torch.zeros(pair_count, dtype=torch.float32) for name in NATIVE_CONTACT_SCALAR_FIELDS}
    )
    native_contacts["frame_counts"] = torch.ones(
        (num_trajectories, num_steps),
        dtype=torch.int64,
    )
    native_contacts["shape_ids_0"] = frame_ids + 10
    native_contacts["shape_ids_1"] = frame_ids + 100
    native_contacts["body_ids_0"] = frame_ids + 20
    native_contacts["body_ids_1"].fill_(-1)
    native_contacts["body_slots_0"] = frame_ids.remainder(4)
    native_contacts["body_slots_1"].fill_(-1)
    native_contacts["world_ids"] = trajectory_ids.repeat_interleave(num_steps)
    native_contacts["separation"] = -(frame_ids.to(torch.float32) + 1.0) * 0.01
    native_contacts["normal_w"][:, 2] = 1.0
    native_contacts["points_0_w"][:, 0] = frame_ids.to(torch.float32)
    native_contacts["points_1_w"][:, 0] = frame_ids.to(torch.float32) + 0.5
    native_contacts["surface_points_0_w"].copy_(native_contacts["points_0_w"])
    native_contacts["surface_points_1_w"].copy_(native_contacts["points_1_w"])

    states = torch.arange(
        trajectory_start * num_steps * 4,
        (trajectory_start + num_trajectories) * num_steps * 4,
        dtype=torch.float32,
    ).reshape(num_trajectories, num_steps, 4)
    root_body_q = torch.zeros((num_trajectories, num_steps, 7), dtype=torch.float32)
    root_body_q[..., 3] = 1.0
    return {
        "states": states,
        "next_states": states + 1.0,
        "joint_f": torch.zeros((num_trajectories, num_steps, 2), dtype=torch.float32),
        "root_body_q": root_body_q,
        "root_body_qd": torch.zeros((num_trajectories, num_steps, 6), dtype=torch.float32),
        "gravity_dir": torch.zeros((num_trajectories, num_steps, 3), dtype=torch.float32),
        "contacts": {},
        "native_contacts": native_contacts,
        "trajectory_context": {
            "state_world_id": trajectory_ids.clone(),
            "root_world_id": trajectory_ids.clone(),
            "contact_world_id": trajectory_ids.clone(),
        },
    }
