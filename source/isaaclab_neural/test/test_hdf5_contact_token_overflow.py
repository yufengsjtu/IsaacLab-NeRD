# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for HDF5 contact-token rollout serialization."""

from __future__ import annotations

import h5py
import pytest
import torch
from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM
from isaaclab_neural.data.hdf5 import append_rollouts_to_hdf5, write_rollouts_to_hdf5


def _token_rollouts(*, num_envs: int = 4, trajectory_length: int = 3, max_tokens: int = 5) -> dict:
    world_ids = torch.arange(num_envs, dtype=torch.long)
    return {
        "states": torch.zeros((num_envs, trajectory_length, 2)),
        "next_states": torch.zeros((num_envs, trajectory_length, 2)),
        "joint_f": torch.zeros((num_envs, trajectory_length, 1)),
        "root_body_q": torch.zeros((num_envs, trajectory_length, 7)),
        "root_body_qd": torch.zeros((num_envs, trajectory_length, 6)),
        "gravity_dir": torch.zeros((num_envs, trajectory_length, 3)),
        "contact_token_body_ids": torch.full(
            (num_envs, trajectory_length, max_tokens), -1, dtype=torch.long
        ),
        "contact_token_world_ids": torch.full(
            (num_envs, trajectory_length, max_tokens), -1, dtype=torch.long
        ),
        "trajectory_context": {
            "state_world_id": world_ids,
            "root_world_id": world_ids,
            "contact_world_id": world_ids,
        },
        "contacts": {
            "contact_tokens": torch.zeros((num_envs, trajectory_length, max_tokens, CONTACT_TOKEN_DIM)),
            # Per-step scalar overflow counts are intentionally rank-2 [N, T].
            "contact_token_overflow": torch.zeros((num_envs, trajectory_length), dtype=torch.long),
        },
    }


def test_write_and_append_accept_2d_contact_token_overflow(tmp_path) -> None:
    path = tmp_path / "tokens.hdf5"
    write_rollouts_to_hdf5(path, _token_rollouts(), env_name="Test-Env")
    append_rollouts_to_hdf5(path, _token_rollouts(), env_name="Test-Env")

    with h5py.File(path, "r") as handle:
        overflow = handle["data"]["contact_token_overflow"]
        assert overflow.shape == (8, 3)
        assert overflow.dtype == "int64"
        assert handle["data"].attrs["contact_token_frame"] == "world_v1"


def test_contact_token_rollout_requires_root_velocity(tmp_path) -> None:
    path = tmp_path / "tokens.hdf5"
    rollouts = _token_rollouts()
    del rollouts["root_body_qd"]

    with pytest.raises(ValueError, match="root_body_qd"):
        write_rollouts_to_hdf5(path, rollouts, env_name="Test-Env")


def test_contact_token_rollout_rejects_float_identity_fields(tmp_path) -> None:
    rollouts = _token_rollouts()
    rollouts["contact_token_world_ids"] = rollouts["contact_token_world_ids"].float()

    with pytest.raises(ValueError, match="must use an integer dtype"):
        write_rollouts_to_hdf5(tmp_path / "tokens.hdf5", rollouts, env_name="Test-Env")


def test_contact_token_rollout_rejects_missing_valid_owner_ids(tmp_path) -> None:
    rollouts = _token_rollouts()
    rollouts["contacts"]["contact_tokens"][0, 0, 0, 0] = 1.0

    with pytest.raises(ValueError, match="nonnegative owner"):
        write_rollouts_to_hdf5(tmp_path / "tokens.hdf5", rollouts, env_name="Test-Env")


def test_contact_token_rollout_rejects_owner_world_row_mismatch(tmp_path) -> None:
    rollouts = _token_rollouts()
    rollouts["contacts"]["contact_tokens"][0, 0, 0, 0] = 1.0
    rollouts["contact_token_body_ids"][0, 0, 0] = 1
    rollouts["contact_token_world_ids"][0, 0, 0] = 1

    with pytest.raises(ValueError, match="do not match their trajectory rows"):
        write_rollouts_to_hdf5(tmp_path / "tokens.hdf5", rollouts, env_name="Test-Env")


def test_rollout_rejects_data_context_key_overlap(tmp_path) -> None:
    rollouts = _token_rollouts()
    rollouts["trajectory_context"] = {"states": torch.zeros(4, dtype=torch.long)}

    with pytest.raises(ValueError, match="keys overlap"):
        write_rollouts_to_hdf5(tmp_path / "tokens.hdf5", rollouts, env_name="Test-Env")
