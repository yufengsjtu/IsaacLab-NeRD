# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the explicit native15 representation with robot self-collisions."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import torch
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_REPRESENTATION_ACTIVE15_SELF,
    CONTACT_REPRESENTATION_RAW15_SELF,
    CONTACT_TOKEN_SELF_COLLISION_FIELD,
    CONTACT_TOKEN_SELF_COLLISION_SCHEMA,
)
from isaaclab_neural.contacts.native15_self_contact_encoder import (
    Active15SelfContactEncoder,
    Raw15SelfContactEncoder,
)
from isaaclab_neural.data.datasets import TrajectoryDataset
from isaaclab_neural.data.hdf5 import append_rollouts_to_hdf5, write_rollouts_to_hdf5


def _extractor(encoder_type, max_tokens: int = 8):
    model = SimpleNamespace(
        body_count=2,
        body_world=torch.tensor([0, 0], dtype=torch.int32),
        body_com=torch.zeros(2, 3),
        shape_margin=torch.tensor([0.01, 0.01, 0.01]),
    )
    return encoder_type(
        model=model,
        primary_body_mask=torch.tensor([True, True]),
        shape_body_torch=torch.tensor([-1, 0, 1]),
        bodies_per_env=2,
        num_envs=1,
        max_contact_tokens=max_tokens,
        device="cpu",
    )


def _state() -> SimpleNamespace:
    half_sqrt = math.sqrt(0.5)
    return SimpleNamespace(
        body_q=torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0, 0.0, half_sqrt, half_sqrt],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        body_qd=torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def _raw_contacts() -> dict[str, torch.Tensor]:
    return {
        "shape0": torch.tensor([1, 1, 1]),
        "shape1": torch.tensor([0, 2, 0]),
        "point0_world": torch.tensor(
            [
                [1.0, 3.0, 0.0],
                [1.0, 3.0, 0.0],
                [1.0, 3.0, 0.0],
            ]
        ),
        "point1_world": torch.tensor(
            [
                [1.0, 3.0, 0.02],
                [1.0, 3.0, 0.02],
                [1.0, 3.0, 0.10],
            ]
        ),
        "normal": torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        "thickness0": torch.tensor([0.03, 0.03, 0.03]),
        "thickness1": torch.tensor([0.02, 0.02, 0.02]),
    }


def _rollouts() -> dict:
    num_envs, steps, capacity = 2, 3, 4
    world_ids = torch.arange(num_envs, dtype=torch.long)
    tokens = torch.zeros(num_envs, steps, capacity, 17)
    tokens[0, 0, 0, 0] = 1.0
    body_ids = torch.full((num_envs, steps, capacity), -1, dtype=torch.long)
    body_ids[0, 0, 0] = 0
    token_world_ids = torch.full_like(body_ids, -1)
    token_world_ids[0, 0, 0] = 0
    self_collision = torch.zeros(num_envs, steps, capacity, dtype=torch.bool)
    self_collision[0, 0, 0] = True
    return {
        "states": torch.zeros(num_envs, steps, 2),
        "next_states": torch.zeros(num_envs, steps, 2),
        "joint_f": torch.zeros(num_envs, steps, 1),
        "root_body_q": torch.zeros(num_envs, steps, 7),
        "root_body_qd": torch.zeros(num_envs, steps, 6),
        "gravity_dir": torch.zeros(num_envs, steps, 3),
        "contact_token_body_ids": body_ids,
        "contact_token_world_ids": token_world_ids,
        "trajectory_context": {
            "state_world_id": world_ids,
            "root_world_id": world_ids,
            "contact_world_id": world_ids,
        },
        "contacts": {
            "contact_tokens": tokens,
            "contact_token_overflow": torch.zeros(num_envs, steps, dtype=torch.long),
            CONTACT_TOKEN_SELF_COLLISION_FIELD: self_collision,
        },
    }


def test_active15_self_adds_two_owner_directed_rows() -> None:
    encoder = _extractor(Active15SelfContactEncoder)
    tokens = encoder.encode(_raw_contacts(), _state())
    valid = tokens[0, :, 0] > 0.5
    active = tokens[0, valid]
    self_collision = encoder.last_self_collision

    assert active.shape == (3, 17)
    assert self_collision is not None
    assert self_collision.shape == (1, 8)
    assert self_collision[0, valid].sum().item() == 2
    assert not self_collision[0, ~valid].any()

    self_rows = active[self_collision[0, valid]]
    self_rows = self_rows[torch.argsort(self_rows[:, 1])]
    torch.testing.assert_close(self_rows[:, 1], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(self_rows[0, 2:5], torch.tensor([1.0, 0.0, 0.0]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(self_rows[0, 5:8], torch.tensor([1.0, 0.0, 0.02]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(self_rows[0, 8:11], torch.tensor([0.0, 0.0, -1.0]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(self_rows[0, 12:15], torch.tensor([0.0, -1.0, 0.0]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(self_rows[0, 15:17], torch.tensor([0.03, 0.02]))
    torch.testing.assert_close(self_rows[1, 2:5], torch.tensor([1.0, 3.0, 0.02]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(self_rows[1, 5:8], torch.tensor([1.0, 3.0, 0.0]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(self_rows[1, 8:11], torch.tensor([0.0, 0.0, 1.0]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(self_rows[1, 12:15], torch.tensor([-1.0, 0.0, 0.0]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(self_rows[1, 15:17], torch.tensor([0.02, 0.03]))
    torch.testing.assert_close(self_rows[:, 11], torch.tensor([-0.01, -0.01]), atol=1.0e-6, rtol=0.0)
    assert encoder.last_overflow.item() == 0


def test_raw15_self_keeps_inactive_external_and_active15_self_filters_it() -> None:
    raw = _raw_contacts()
    raw_encoder = _extractor(Raw15SelfContactEncoder)
    active_encoder = _extractor(Active15SelfContactEncoder)

    raw_tokens = raw_encoder.encode(raw, _state())
    active_tokens = active_encoder.encode(raw, _state())

    assert (raw_tokens[..., 0] > 0.5).sum().item() == 4
    assert (active_tokens[..., 0] > 0.5).sum().item() == 3
    assert raw_encoder.last_self_collision is not None
    assert raw_encoder.last_self_collision.sum().item() == 2
    assert active_encoder.last_self_collision is not None
    assert active_encoder.last_self_collision.sum().item() == 2


def test_native15_self_sidecar_metadata_roundtrip_and_padding_validation(tmp_path: Path) -> None:
    path = tmp_path / "raw15_self.hdf5"
    write_rollouts_to_hdf5(
        path,
        _rollouts(),
        env_name="Anymal-C-Rough-Native-Raw15Self",
        contact_representation=CONTACT_REPRESENTATION_RAW15_SELF,
    )
    append_rollouts_to_hdf5(
        path,
        _rollouts(),
        env_name="Anymal-C-Rough-Native-Raw15Self",
        contact_representation=CONTACT_REPRESENTATION_RAW15_SELF,
    )

    with h5py.File(path, "r") as handle:
        data = handle["data"]
        assert data.attrs["contact_representation"] == CONTACT_REPRESENTATION_RAW15_SELF
        assert data.attrs["contact_schema"] == "raw15_self_owner_body_v1"
        assert data.attrs["contact_selection"] == "newton_raw_candidates_with_directed_robot_self_v1"
        assert data.attrs["contact_token_self_collision_schema"] == CONTACT_TOKEN_SELF_COLLISION_SCHEMA
        assert data[CONTACT_TOKEN_SELF_COLLISION_FIELD].dtype.kind == "b"

    dataset = TrajectoryDataset(
        path,
        sample_sequence_length=2,
        expected_contact_representation=CONTACT_REPRESENTATION_RAW15_SELF,
    )
    assert dataset[0][CONTACT_TOKEN_SELF_COLLISION_FIELD].dtype == torch.bool

    invalid = _rollouts()
    invalid["contacts"][CONTACT_TOKEN_SELF_COLLISION_FIELD][0, 0, 1] = True
    with pytest.raises(ValueError, match="padded"):
        write_rollouts_to_hdf5(
            tmp_path / "invalid.hdf5",
            invalid,
            env_name="Anymal-C-Rough-Native-Raw15Self",
            contact_representation=CONTACT_REPRESENTATION_RAW15_SELF,
        )


def test_native15_self_representation_constants_are_distinct() -> None:
    assert CONTACT_REPRESENTATION_ACTIVE15_SELF == "active15_self_tokens"
    assert CONTACT_REPRESENTATION_RAW15_SELF == "raw15_self_tokens"
