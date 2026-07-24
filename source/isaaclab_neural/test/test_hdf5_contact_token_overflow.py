# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for HDF5 contact-token rollout serialization."""

from __future__ import annotations

import h5py
import torch
from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM
from isaaclab_neural.data.hdf5 import append_rollouts_to_hdf5, write_rollouts_to_hdf5


def _token_rollouts(*, num_envs: int = 4, trajectory_length: int = 3, max_tokens: int = 5) -> dict:
    return {
        "states": torch.zeros((num_envs, trajectory_length, 2)),
        "next_states": torch.zeros((num_envs, trajectory_length, 2)),
        "joint_f": torch.zeros((num_envs, trajectory_length, 1)),
        "root_body_q": torch.zeros((num_envs, trajectory_length, 7)),
        "gravity_dir": torch.zeros((num_envs, trajectory_length, 3)),
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
