# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native-only HDF5 schema and append regression tests."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from isaaclab_neural.data.hdf5 import append_rollouts_to_hdf5, write_rollouts_to_hdf5
from native_hdf5_samples import make_native_rollouts


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("contact_masks", torch.zeros((2, 3, 1), dtype=torch.bool)),
        ("contact_tokens", torch.zeros((2, 3, 1, 14), dtype=torch.float32)),
    ],
)
def test_native_schema_rejects_baked_contact_fields(tmp_path, field_name, field_value) -> None:
    rollouts = make_native_rollouts()
    rollouts["contacts"][field_name] = field_value

    with pytest.raises(ValueError, match="baked contact fields are incompatible"):
        write_rollouts_to_hdf5(tmp_path / "native.hdf5", rollouts, env_name="Test-Env")


def test_native_rollout_rejects_mismatched_trajectory_world_identity(tmp_path) -> None:
    rollouts = make_native_rollouts()
    rollouts["trajectory_context"]["contact_world_id"][0] = 7

    with pytest.raises(ValueError, match="State, root, and contact trajectory rows"):
        write_rollouts_to_hdf5(tmp_path / "native.hdf5", rollouts, env_name="Test-Env")


def test_native_rollout_rejects_pair_world_identity_mismatch(tmp_path) -> None:
    rollouts = make_native_rollouts()
    rollouts["native_contacts"]["world_ids"][0] = 7

    with pytest.raises(ValueError, match="pair world ids do not match"):
        write_rollouts_to_hdf5(tmp_path / "native.hdf5", rollouts, env_name="Test-Env")


def test_native_rollout_requires_root_velocity(tmp_path) -> None:
    rollouts = make_native_rollouts()
    del rollouts["root_body_qd"]

    with pytest.raises(ValueError, match="root_body_qd"):
        write_rollouts_to_hdf5(tmp_path / "native.hdf5", rollouts, env_name="Test-Env")


def _assert_hdf5_groups_equal(actual: h5py.Group, expected: h5py.Group) -> None:
    assert set(actual.keys()) == set(expected.keys())
    assert set(actual.attrs.keys()) == set(expected.attrs.keys())
    for name in actual.attrs:
        np.testing.assert_equal(actual.attrs[name], expected.attrs[name])
    for name in actual:
        actual_item = actual[name]
        expected_item = expected[name]
        assert isinstance(actual_item, h5py.Group) == isinstance(expected_item, h5py.Group)
        if isinstance(actual_item, h5py.Group):
            _assert_hdf5_groups_equal(actual_item, expected_item)
        else:
            np.testing.assert_equal(actual_item[()], expected_item[()])


def test_native_append_matches_single_write(tmp_path) -> None:
    appended_path = tmp_path / "appended.hdf5"
    single_write_path = tmp_path / "single_write.hdf5"

    append_rollouts_to_hdf5(
        appended_path,
        make_native_rollouts(num_trajectories=2, trajectory_start=0),
        env_name="Test-Env",
    )
    append_rollouts_to_hdf5(
        appended_path,
        make_native_rollouts(num_trajectories=2, trajectory_start=2),
        env_name="Test-Env",
    )
    write_rollouts_to_hdf5(
        single_write_path,
        make_native_rollouts(num_trajectories=4, trajectory_start=0),
        env_name="Test-Env",
    )

    with h5py.File(appended_path, "r") as appended, h5py.File(single_write_path, "r") as single:
        _assert_hdf5_groups_equal(appended, single)
