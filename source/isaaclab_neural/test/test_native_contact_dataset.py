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


@pytest.mark.parametrize(
    "contact_filter,has_contacts",
    [("solver_active", False), ("solver_active_movable", False), ("solver_active_movable", True)],
)
def test_filtered_empty_ranges_skip_geometry_reads(tmp_path, monkeypatch, contact_filter, has_contacts):
    from isaaclab_neural.data.native_contacts import NativeContactProjection, read_and_project_frames

    path = tmp_path / "native.hdf5"
    rollouts = _native_rollouts()
    pairs = rollouts["native_contacts"]
    if contact_filter == "solver_active":
        pairs["separation"].fill_(0.1)
    fixed = (0,) if has_contacts else (0, 1)
    write_rollouts_to_hdf5(path, rollouts, env_name="Test-Env")
    expected = project_native_pairs(
        {key: value for key, value in pairs.items() if key != "frame_counts"},
        num_frames=2,
        frame_ids=torch.zeros(2, dtype=torch.long),
        representation="contact_tokens",
        max_contact_tokens=4,
        packing_policy="body_round_robin_pair_atomic",
        contact_filter=contact_filter,
        fixed_body_slots=fixed,
    )
    geometry_reads = []
    original = h5py.Dataset.__getitem__

    def read(dataset, *args, **kwargs):
        if dataset.name.rsplit("/", 1)[-1] in NATIVE_CONTACT_VECTOR_FIELDS:
            geometry_reads.append(dataset.name)
        return original(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    with h5py.File(path, "r") as file:
        actual = read_and_project_frames(
            file["contacts/native_pairs"],
            np.array([0, 1, 0], dtype=np.int64),
            NativeContactProjection("contact_tokens", 4, "body_round_robin_pair_atomic", contact_filter, fixed),
        )
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name][[0, 1, 0]].numpy())
    assert bool(geometry_reads) == has_contacts


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


@pytest.mark.parametrize("both_sides", [False, True])
def test_movable_filter_only_removes_immovable_pairs(both_sides):
    pairs = {
        key: value
        for key, value in _native_rollouts(both_sides_on_first_pair=both_sides)["native_contacts"].items()
        if key != "frame_counts"
    }
    original = {key: value.clone() for key, value in pairs.items()}
    projected = project_native_pairs(
        pairs,
        num_frames=1,
        representation="contact_tokens",
        packing_policy="body_round_robin_pair_atomic",
        max_contact_tokens=4,
        contact_filter="solver_active_movable",
        fixed_body_slots=(0,),
    )
    assert int((projected["contact_tokens"][..., 0] > 0.5).sum()) == (3 if both_sides else 1)
    assert int(projected["contact_token_overflow"].sum()) == 0
    for key in pairs:
        torch.testing.assert_close(pairs[key], original[key])
    pairs["separation"][1] = 0.1
    inactive = project_native_pairs(
        pairs,
        num_frames=1,
        representation="contact_tokens",
        packing_policy="body_round_robin_pair_atomic",
        max_contact_tokens=4,
        contact_filter="solver_active_movable",
        fixed_body_slots=(0,),
    )
    assert int((inactive["contact_tokens"][..., 0] > 0.5).sum()) == (2 if both_sides else 0)
    if both_sides:
        pairs["body_slots_1"][0] = -1
        untracked = project_native_pairs(
            pairs,
            num_frames=1,
            representation="contact_tokens",
            packing_policy="body_round_robin_pair_atomic",
            max_contact_tokens=4,
            contact_filter="solver_active_movable",
            fixed_body_slots=(0,),
        )
        assert int((untracked["contact_tokens"][..., 0] > 0.5).sum()) == 1
    with pytest.raises(ValueError, match="fixed.*slots|mobility"):
        project_native_pairs(
            pairs,
            num_frames=1,
            representation="contact_tokens",
            packing_policy="body_round_robin_pair_atomic",
            max_contact_tokens=4,
            contact_filter="solver_active_movable",
        )


def test_movable_filter_dataset_requires_recorded_mobility(tmp_path):
    import json

    from isaaclab_neural.data.datasets import create_trajectory_dataset

    path = tmp_path / "native.hdf5"
    write_rollouts_to_hdf5(path, _native_rollouts(), env_name="Test-Env")
    kwargs = dict(
        hdf5_dataset_path=path,
        sample_sequence_length=2,
        contact_representation="contact_tokens",
        max_contact_tokens=4,
        contact_packing_policy="body_round_robin_pair_atomic",
        contact_filter="solver_active_movable",
    )
    with pytest.raises(ValueError, match="mobility|fixed.*slots"):
        create_trajectory_dataset(load_mode="lazy", **kwargs)
    with h5py.File(path, "a") as file:
        file["data"].attrs["keyboard_contract"] = json.dumps(
            {"contact_filter": "solver_active_movable", "contact_fixed_body_slots": [0]}
        )
    for mode in ("lazy", "eager"):
        dataset = create_trajectory_dataset(load_mode=mode, **kwargs)
        assert int((dataset[0]["contact_tokens"][..., 0] > 0.5).sum()) == 1
        if hasattr(dataset, "close"):
            dataset.close()


def test_fixed_body_classification_inherits_parent_mobility():
    from types import SimpleNamespace

    from isaaclab_neural.contacts.newton_contact_adapter import find_immovable_primary_slots

    class Array:
        def __init__(self, values):
            self.values = np.asarray(values)

        def numpy(self):
            return self.values

    model = SimpleNamespace(
        body_count=6,
        body_flags=Array([1] * 6),
        joint_parent=Array([-1, 0, 1, -1, 3, 4]),
        joint_child=Array(range(6)),
        joint_qd_start=Array([0, 0, 1, 1, 1, 2, 2]),
    )
    assert find_immovable_primary_slots(model, torch.tensor([[0, 1, 2], [3, 4, 5]])) == (0,)


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
