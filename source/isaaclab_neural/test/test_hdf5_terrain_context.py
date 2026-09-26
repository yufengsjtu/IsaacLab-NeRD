# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for terrain provenance in NeRD HDF5 datasets."""

import h5py
import numpy as np
import pytest
import torch
from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM
from isaaclab_neural.data import TrajectoryDataset, append_rollouts_to_hdf5, write_rollouts_to_hdf5
from native_hdf5_samples import make_native_rollouts


def _rollouts(num_trajectories: int = 2, num_steps: int = 3):
    rollouts = make_native_rollouts(
        num_trajectories=num_trajectories,
        num_steps=num_steps,
    )
    rollouts["trajectory_context"].update(
        {
            "source_env_id": torch.arange(num_trajectories, dtype=torch.int64),
            "terrain_level": torch.arange(num_trajectories, dtype=torch.int64),
            "terrain_type": torch.arange(num_trajectories, dtype=torch.int64) + 4,
            "env_origin": torch.arange(num_trajectories * 3, dtype=torch.float32).reshape(num_trajectories, 3),
        }
    )
    return rollouts


def _terrain_context(seed: int = 40):
    return {
        "schema_version": 1,
        "seed": seed,
        "config_sha256": "config",
        "mesh_sha256": "mesh",
        "origins_sha256": "origins",
        "num_rows": 10,
        "num_cols": 20,
    }


def test_hdf5_round_trip_preserves_terrain_context(tmp_path):
    dataset_path = tmp_path / "dataset.hdf5"
    write_rollouts_to_hdf5(dataset_path, _rollouts(), "rough", terrain_context=_terrain_context())

    dataset = TrajectoryDataset(dataset_path, sample_sequence_length=2)
    sample = dataset[0]

    assert sample["states"].shape == (2, 4)
    assert sample["source_env_id"].dtype == torch.int64
    assert sample["root_body_qd"].shape == (2, 6)
    assert sample["terrain_type"].item() == 4
    assert torch.equal(sample["env_origin"], torch.tensor([0.0, 1.0, 2.0]))
    with h5py.File(dataset_path, "r") as dataset_file:
        assert dataset_file["context/terrain"].attrs["seed"] == 40
        origins = dataset_file["context/trajectories/env_origin"]
        assert isinstance(origins, h5py.Dataset)
        assert origins.shape == (2, 3)


@pytest.mark.parametrize("load_mode", ["eager", "lazy"])
@pytest.mark.parametrize("path", ["context", "context/trajectories"])
@pytest.mark.parametrize("object_type", ["dataset", "datatype"])
def test_trajectory_loader_rejects_non_group_context(tmp_path, load_mode, path, object_type):
    from isaaclab_neural.data import create_trajectory_dataset

    dataset_path = tmp_path / "malformed.hdf5"
    write_rollouts_to_hdf5(dataset_path, _rollouts(), "rough", terrain_context=_terrain_context())
    with h5py.File(dataset_path, "a") as file:
        del file[path]
        if object_type == "dataset":
            file.create_dataset(path, data=[0])
        else:
            file[path] = np.dtype("float32")
    with pytest.raises(ValueError, match="must be an HDF5 group"):
        create_trajectory_dataset(load_mode=load_mode, hdf5_dataset_path=dataset_path, sample_sequence_length=2)


def test_append_rejects_mixed_terrain_context(tmp_path):
    dataset_path = tmp_path / "dataset.hdf5"
    append_rollouts_to_hdf5(dataset_path, _rollouts(), "rough", terrain_context=_terrain_context())

    with pytest.raises(ValueError, match="Terrain context"):
        append_rollouts_to_hdf5(dataset_path, _rollouts(), "rough", terrain_context=_terrain_context(seed=41))

    append_rollouts_to_hdf5(dataset_path, _rollouts(), "rough", terrain_context=_terrain_context())
    with h5py.File(dataset_path, "r") as dataset_file:
        states = dataset_file["data/states"]
        terrain_level = dataset_file["context/trajectories/terrain_level"]
        assert isinstance(states, h5py.Dataset)
        assert isinstance(terrain_level, h5py.Dataset)
        assert states.shape[0] == 4
        assert terrain_level.shape[0] == 4


def test_history_and_rollout_window_stays_within_trajectory(tmp_path):
    dataset_path = tmp_path / "dataset.hdf5"
    write_rollouts_to_hdf5(
        dataset_path,
        _rollouts(num_trajectories=2, num_steps=20),
        "rough",
        terrain_context=_terrain_context(),
    )

    dataset = TrajectoryDataset(dataset_path, sample_sequence_length=19)

    assert len(dataset) == 4
    assert dataset.mapping_index2traj.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_rank_sharded_contact_tokens_keep_matching_trajectory_context(tmp_path):
    dataset_path = tmp_path / "tokens.hdf5"
    rollouts = _rollouts(num_trajectories=4)
    write_rollouts_to_hdf5(dataset_path, rollouts, "rough-tokens", terrain_context=_terrain_context())

    dataset = TrajectoryDataset(
        dataset_path,
        sample_sequence_length=2,
        rank=1,
        world_size=2,
        contact_representation="contact_tokens",
        max_contact_tokens=2,
        contact_packing_policy="body_round_robin_pair_atomic",
    )
    sample = dataset[0]

    assert dataset.global_trajectory_indices.tolist() == [1, 3]
    assert sample["contact_tokens"].shape == (2, 2, CONTACT_TOKEN_DIM)
    assert int(sample["contact_tokens"][..., 0].sum()) == 2
    assert sample["contact_token_world_ids"][:, 0].tolist() == [1, 1]
    assert sample["source_env_id"].item() == 1
    assert sample["terrain_type"].item() == 5
