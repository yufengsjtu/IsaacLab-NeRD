# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for terrain provenance in NeRD HDF5 datasets."""

import h5py
import pytest
import torch

from isaaclab_neural.data import TrajectoryDataset, append_rollouts_to_hdf5, write_rollouts_to_hdf5


def _rollouts(num_trajectories: int = 2, num_steps: int = 3):
    return {
        "states": torch.arange(num_trajectories * num_steps * 4, dtype=torch.float32).reshape(
            num_trajectories, num_steps, 4
        ),
        "next_states": torch.zeros(num_trajectories, num_steps, 4),
        "joint_f": torch.zeros(num_trajectories, num_steps, 2),
        "contacts": {
            "contact_masks": torch.zeros(num_trajectories, num_steps, 2, dtype=torch.bool),
            "contact_depths": torch.zeros(num_trajectories, num_steps, 2),
        },
        "trajectory_context": {
            "source_env_id": torch.arange(num_trajectories, dtype=torch.int64),
            "terrain_level": torch.arange(num_trajectories, dtype=torch.int64),
            "terrain_type": torch.arange(num_trajectories, dtype=torch.int64) + 4,
            "env_origin": torch.arange(num_trajectories * 3, dtype=torch.float32).reshape(num_trajectories, 3),
        },
    }


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
    assert sample["terrain_type"].item() == 4
    assert torch.equal(sample["env_origin"], torch.tensor([0.0, 1.0, 2.0]))
    with h5py.File(dataset_path, "r") as dataset_file:
        assert dataset_file["context"]["terrain"].attrs["seed"] == 40
        assert dataset_file["context"]["trajectories"]["env_origin"].shape == (2, 3)


def test_append_rejects_mixed_terrain_context(tmp_path):
    dataset_path = tmp_path / "dataset.hdf5"
    append_rollouts_to_hdf5(dataset_path, _rollouts(), "rough", terrain_context=_terrain_context())

    with pytest.raises(ValueError, match="Terrain context"):
        append_rollouts_to_hdf5(dataset_path, _rollouts(), "rough", terrain_context=_terrain_context(seed=41))

    append_rollouts_to_hdf5(dataset_path, _rollouts(), "rough", terrain_context=_terrain_context())
    with h5py.File(dataset_path, "r") as dataset_file:
        assert dataset_file["data"]["states"].shape[0] == 4
        assert dataset_file["context"]["trajectories"]["terrain_level"].shape[0] == 4


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
