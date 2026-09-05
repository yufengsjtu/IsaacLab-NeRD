# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for unified ragged native-contact datasets."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_TOKEN_DIM,
    NATIVE_CONTACT_INTEGER_FIELDS,
    NATIVE_CONTACT_SCALAR_FIELDS,
    NATIVE_CONTACT_VECTOR_FIELDS,
)
from isaaclab_neural.contacts.native_pair_projector import project_native_pairs
from isaaclab_neural.data.datasets import TrajectoryDataset
from isaaclab_neural.data.hdf5 import append_rollouts_to_hdf5, write_rollouts_to_hdf5
from isaaclab_neural.eval.contact_reconstruction_diagnostic import load_dataset_batch


def _native_rollouts(*, both_sides_on_first_pair: bool = False) -> dict:
    pair_count = 2
    native_contacts = {name: torch.zeros(pair_count, dtype=torch.int64) for name in NATIVE_CONTACT_INTEGER_FIELDS}
    native_contacts.update(
        {name: torch.zeros((pair_count, 3), dtype=torch.float32) for name in NATIVE_CONTACT_VECTOR_FIELDS}
    )
    native_contacts.update(
        {name: torch.zeros(pair_count, dtype=torch.float32) for name in NATIVE_CONTACT_SCALAR_FIELDS}
    )
    native_contacts["frame_counts"] = torch.tensor([[2, 0]], dtype=torch.int64)
    native_contacts["shape_ids_0"] = torch.tensor([2, 3], dtype=torch.int64)
    native_contacts["shape_ids_1"] = torch.tensor([8, 9], dtype=torch.int64)
    native_contacts["body_ids_0"] = torch.tensor([10, 11], dtype=torch.int64)
    native_contacts["body_ids_1"] = torch.tensor([12 if both_sides_on_first_pair else -1, -1], dtype=torch.int64)
    native_contacts["body_slots_0"] = torch.tensor([0, 1], dtype=torch.int64)
    native_contacts["body_slots_1"] = torch.tensor(
        [2 if both_sides_on_first_pair else -1, -1],
        dtype=torch.int64,
    )
    native_contacts["world_ids"] = torch.zeros(pair_count, dtype=torch.int64)
    native_contacts["separation"] = torch.tensor([-0.2, -0.1], dtype=torch.float32)
    native_contacts["normal_w"][:, 2] = 1.0
    native_contacts["surface_points_0_w"][:, 0] = torch.tensor([1.0, 2.0])
    native_contacts["surface_points_1_w"][:, 0] = torch.tensor([1.5, 2.5])
    native_contacts["points_0_w"].copy_(native_contacts["surface_points_0_w"])
    native_contacts["points_1_w"].copy_(native_contacts["surface_points_1_w"])
    native_contacts["points_1_w"][:, 2] = native_contacts["separation"]

    root_q = torch.zeros((1, 2, 7), dtype=torch.float32)
    root_q[..., 3] = 1.0
    world_ids = torch.zeros(1, dtype=torch.int64)
    return {
        "states": torch.zeros((1, 2, 2), dtype=torch.float32),
        "next_states": torch.zeros((1, 2, 2), dtype=torch.float32),
        "joint_f": torch.zeros((1, 2, 1), dtype=torch.float32),
        "root_body_q": root_q,
        "root_body_qd": torch.zeros((1, 2, 6), dtype=torch.float32),
        "gravity_dir": torch.zeros((1, 2, 3), dtype=torch.float32),
        "contacts": {},
        "native_contacts": native_contacts,
        "trajectory_context": {
            "state_world_id": world_ids,
            "root_world_id": world_ids,
            "contact_world_id": world_ids,
        },
    }


def test_native_contacts_are_compact_and_appendable(tmp_path) -> None:
    path = tmp_path / "native.hdf5"
    rollouts = _native_rollouts()
    write_rollouts_to_hdf5(path, rollouts, env_name="Test-Env")
    append_rollouts_to_hdf5(path, rollouts, env_name="Test-Env")

    with h5py.File(path, "r") as handle:
        data_group = handle["data"]
        native_group = handle["contacts"]["native_pairs"]
        np.testing.assert_array_equal(native_group["frame_offsets"][()], np.array([0, 2, 2, 4, 4]))
        assert native_group["normal_w"].shape == (4, 3)
        assert native_group["normal_w"].compression == "lzf"
        assert data_group.attrs["contact_storage_schema"] == "native_pairs_csr_v1"
        assert "contact_tokens" not in data_group
        assert "contact_masks" not in data_group


def test_one_native_dataset_projects_to_flat_and_tokens(tmp_path) -> None:
    path = tmp_path / "native.hdf5"
    write_rollouts_to_hdf5(path, _native_rollouts(), env_name="Test-Env")

    flat = TrajectoryDataset(
        path,
        sample_sequence_length=1,
        contact_representation="flat",
        num_contacts_per_env=1,
        contact_packing_policy="penetration_priority",
    )
    tokens = TrajectoryDataset(
        path,
        sample_sequence_length=1,
        contact_representation="contact_tokens",
        max_contact_tokens=4,
        contact_packing_policy="body_round_robin_pair_atomic",
    )

    flat_sample = flat[0]
    token_sample = tokens[0]
    torch.testing.assert_close(flat_sample["contact_depths"], torch.tensor([[-0.2]]))
    assert flat_sample["contact_masks"].tolist() == [[True]]
    assert token_sample["contact_tokens"].shape == (1, 4, CONTACT_TOKEN_DIM)
    assert int(token_sample["contact_tokens"][..., 0].sum()) == 2
    assert int(token_sample["contact_token_overflow"].item()) == 0


def test_trajectory_dataset_keeps_primary_body_q_pose_axes(tmp_path) -> None:
    path = tmp_path / "native.hdf5"
    rollouts = _native_rollouts()
    rollouts["primary_body_q"] = torch.zeros((*rollouts["states"].shape[:2], 17, 7), dtype=torch.float32)
    rollouts["primary_body_q"][..., 6] = 1.0
    write_rollouts_to_hdf5(path, rollouts, env_name="Test-Env")

    dataset = TrajectoryDataset(
        path,
        sample_sequence_length=2,
        contact_representation="contact_tokens",
        max_contact_tokens=4,
        contact_packing_policy="body_round_robin_pair_atomic",
    )
    sample = dataset[0]
    assert sample["primary_body_q"].shape == (2, 17, 7)
    torch.testing.assert_close(sample["primary_body_q"][..., 6], torch.ones(2, 17))


def test_solver_active_filter_projects_only_penetrating_pairs(tmp_path) -> None:
    path = tmp_path / "native.hdf5"
    rollouts = _native_rollouts()
    rollouts["native_contacts"]["separation"] = torch.tensor([0.01, -0.1])
    rollouts["native_contacts"]["points_1_w"][:, 2] = -1.0
    write_rollouts_to_hdf5(path, rollouts, env_name="Test-Env")

    flat = TrajectoryDataset(
        path,
        sample_sequence_length=1,
        contact_representation="flat",
        num_contacts_per_env=2,
        contact_packing_policy="penetration_priority",
        contact_filter="solver_active",
    )
    tokens = TrajectoryDataset(
        path,
        sample_sequence_length=1,
        contact_representation="contact_tokens",
        max_contact_tokens=4,
        contact_packing_policy="body_round_robin_pair_atomic",
        contact_filter="solver_active",
    )

    flat_sample = flat[0]
    token_sample = tokens[0]
    assert flat_sample["contact_masks"].tolist() == [[True, False]]
    torch.testing.assert_close(flat_sample["contact_depths"], torch.tensor([[-0.1, 0.0]]))
    assert int(token_sample["contact_tokens"][..., 0].sum()) == 1
    torch.testing.assert_close(token_sample["contact_tokens"][0, 0, 13], torch.tensor(-0.1))


def test_solver_active_filter_uses_recorded_separation() -> None:
    native_pairs = {
        name: value for name, value in _native_rollouts()["native_contacts"].items() if name != "frame_counts"
    }
    native_pairs["separation"] = torch.tensor([0.0, -1.0e-6])
    native_pairs["points_1_w"][:, 2] = -1.0

    projected = project_native_pairs(
        native_pairs,
        num_frames=1,
        representation="flat",
        packing_policy="penetration_priority",
        flat_capacity=2,
        return_pair_indices=True,
        contact_filter="solver_active",
    )

    assert projected["contact_masks"].tolist() == [[True, False]]
    torch.testing.assert_close(projected["contact_depths"], torch.tensor([[-1.0e-6, 0.0]]))
    assert projected["_pair_indices"].tolist() == [[1, -1]]


def test_token_projection_matches_pair_atomic_overflow_semantics(tmp_path) -> None:
    path = tmp_path / "native.hdf5"
    write_rollouts_to_hdf5(
        path,
        _native_rollouts(both_sides_on_first_pair=True),
        env_name="Test-Env",
    )

    tokens = TrajectoryDataset(
        path,
        sample_sequence_length=1,
        contact_representation="contact_tokens",
        max_contact_tokens=1,
        contact_packing_policy="body_round_robin_pair_atomic",
    )
    sample = tokens[0]

    assert int(sample["contact_tokens"][..., 0].sum()) == 0
    assert int(sample["contact_token_overflow"].item()) == 3


def test_token_round_robin_ranks_are_independent_per_frame() -> None:
    native_pairs = _native_rollouts()["native_contacts"]
    native_pairs = {
        name: torch.cat((value, value), dim=0) for name, value in native_pairs.items() if name != "frame_counts"
    }
    native_pairs["body_ids_0"] = torch.tensor([10, 10, 10, 11])
    native_pairs["body_slots_0"] = torch.tensor([0, 0, 0, 1])
    native_pairs["body_ids_1"].fill_(-1)
    native_pairs["body_slots_1"].fill_(-1)
    native_pairs["world_ids"] = torch.tensor([0, 0, 1, 1])
    native_pairs["separation"] = torch.tensor([-0.4, -0.3, -0.5, -0.1])
    native_pairs["points_1_w"][:, 2] = native_pairs["separation"]

    projected = project_native_pairs(
        native_pairs,
        num_frames=2,
        representation="contact_tokens",
        packing_policy="body_round_robin_pair_atomic",
        max_contact_tokens=1,
    )

    assert projected["contact_token_body_ids"][:, 0].tolist() == [10, 10]


def test_native_schema_rejects_wrong_vector_shape(tmp_path) -> None:
    rollouts = _native_rollouts()
    rollouts["native_contacts"]["normal_w"] = torch.zeros((2, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="normal_w.*shape"):
        write_rollouts_to_hdf5(tmp_path / "native.hdf5", rollouts, env_name="Test-Env")


def test_invalid_force_body_names_do_not_partially_append(tmp_path) -> None:
    path = tmp_path / "native.hdf5"
    rollouts = _native_rollouts()
    rollouts["body_net_contact_forces_w"] = torch.zeros((1, 2, 1, 3), dtype=torch.float32)
    write_rollouts_to_hdf5(
        path,
        rollouts,
        env_name="Test-Env",
        body_net_contact_force_body_names=("base",),
    )

    with pytest.raises(ValueError, match="body-name count"):
        append_rollouts_to_hdf5(
            path,
            rollouts,
            env_name="Test-Env",
            body_net_contact_force_body_names=("base", "extra"),
        )

    with h5py.File(path, "r") as handle:
        assert handle["data"]["states"].shape[0] == 1
        np.testing.assert_array_equal(handle["contacts"]["native_pairs"]["frame_offsets"][()], [0, 2, 2])


def test_contact_diagnostic_projects_native_dataset(tmp_path) -> None:
    path = tmp_path / "native.hdf5"
    write_rollouts_to_hdf5(path, _native_rollouts(), env_name="Test-Env")

    batch = load_dataset_batch(
        str(path),
        trajectory_start=0,
        num_envs=1,
        step=0,
        device="cpu",
        contact_representation="contact_tokens",
        max_contact_tokens=4,
        contact_packing_policy="body_round_robin_pair_atomic",
    )

    assert batch["contact_tokens"].shape == (1, 1, 4, CONTACT_TOKEN_DIM)
    assert int(batch["contact_tokens"][..., 0].sum()) == 2
