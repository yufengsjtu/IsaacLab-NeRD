# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for G1 HDF5 trajectory replay helpers."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from isaaclab_neural.eval.replay_g1_dataset import (
    FLAT_DATASET_GEN_TASK,
    ROUGH_DATASET_GEN_TASK,
    apply_terrain_context,
    clip_and_pad_states,
    infer_replay_task,
    load_replay_batch,
    read_replay_dataset_info,
    sample_trajectory_indices,
)


def _write_dataset(
    path,
    *,
    num_trajectories: int = 4,
    horizon: int = 8,
    state_dim: int = 6,
    env_name: str = "G1-Native-ContactTokens",
    traj_lengths: np.ndarray | None = None,
    include_context: bool = True,
) -> None:
    states = np.arange(num_trajectories * horizon * state_dim, dtype=np.float32).reshape(
        num_trajectories, horizon, state_dim
    )
    with h5py.File(path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        data_group.create_dataset("states", data=states)
        data_group.attrs["env"] = env_name
        data_group.attrs["mode"] = "trajectory"
        if traj_lengths is not None:
            data_group.create_dataset("traj_lengths", data=np.asarray(traj_lengths, dtype=np.int32))
        if include_context:
            trajectory_group = dataset_file.create_group("context").create_group("trajectories")
            trajectory_group.create_dataset(
                "env_origin",
                data=np.arange(num_trajectories * 3, dtype=np.float32).reshape(num_trajectories, 3),
            )
            trajectory_group.create_dataset("terrain_level", data=np.full(num_trajectories, -1, dtype=np.int64))
            trajectory_group.create_dataset("terrain_type", data=np.full(num_trajectories, -1, dtype=np.int64))


def test_infer_replay_task_defaults_to_flat_dataset_gen():
    assert infer_replay_task("G1-Native-ContactTokens") == FLAT_DATASET_GEN_TASK
    assert infer_replay_task("G1-Rough-Native") == ROUGH_DATASET_GEN_TASK
    assert infer_replay_task("G1-Rough-Native", "Isaac-Velocity-Flat-G1") == "Isaac-Velocity-Flat-G1"


def test_sample_trajectory_indices_random_and_explicit():
    rng = np.random.default_rng(0)
    random_indices = sample_trajectory_indices(5, 3, rng)
    assert random_indices.shape == (3,)
    assert np.all((random_indices >= 0) & (random_indices < 5))

    broadcast = sample_trajectory_indices(5, 3, rng, trajectory_indices=[2])
    np.testing.assert_array_equal(broadcast, np.array([2, 2, 2]))

    explicit = sample_trajectory_indices(5, 2, rng, trajectory_indices=[0, 4])
    np.testing.assert_array_equal(explicit, np.array([0, 4]))


def test_sample_trajectory_indices_rejects_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        sample_trajectory_indices(3, 1, np.random.default_rng(0), trajectory_indices=[3])
    with pytest.raises(ValueError, match="Expected 2 trajectory indices"):
        sample_trajectory_indices(5, 2, np.random.default_rng(0), trajectory_indices=[0, 1, 2])


def test_read_and_load_replay_batch(tmp_path):
    dataset_path = tmp_path / "dataset_valid.hdf5"
    _write_dataset(dataset_path, traj_lengths=np.array([8, 5, 8, 3], dtype=np.int32))

    info = read_replay_dataset_info(dataset_path)
    assert info.env_name == "G1-Native-ContactTokens"
    assert info.num_trajectories == 4
    assert info.horizon == 8
    assert info.state_dim == 6
    assert info.traj_lengths is not None
    np.testing.assert_array_equal(info.traj_lengths, [8, 5, 8, 3])

    batch = load_replay_batch(dataset_path, [1, 1, 3])
    assert batch["states"].shape == (3, 8, 6)
    torch.testing.assert_close(batch["states"][0], batch["states"][1])
    np.testing.assert_array_equal(batch["traj_lengths"].numpy(), [5, 5, 3])
    torch.testing.assert_close(batch["env_origin"][2], torch.tensor([9.0, 10.0, 11.0]))
    padded = clip_and_pad_states(batch["states"], batch["traj_lengths"], max_steps=6)
    assert padded.shape == (3, 6, 6)
    torch.testing.assert_close(padded[2, 3], batch["states"][2, 2])


def test_clip_and_pad_states_holds_last_valid_frame():
    states = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    padded = clip_and_pad_states(states, np.array([6, 2]), start_step=0, max_steps=4)

    assert padded.shape == (2, 4, 3)
    torch.testing.assert_close(padded[0], states[0, :4])
    torch.testing.assert_close(padded[1, 0], states[1, 0])
    torch.testing.assert_close(padded[1, 1], states[1, 1])
    torch.testing.assert_close(padded[1, 2], states[1, 1])
    torch.testing.assert_close(padded[1, 3], states[1, 1])


def test_clip_and_pad_states_start_step_and_errors():
    states = torch.zeros(1, 5, 2)
    sliced = clip_and_pad_states(states, None, start_step=2, max_steps=10)
    assert sliced.shape == (1, 3, 2)
    with pytest.raises(ValueError, match="no frames"):
        clip_and_pad_states(states, np.array([2]), start_step=2)


def test_apply_terrain_context_skips_sentinel_levels_on_flat_terrain():
    terrain = SimpleNamespace(
        device=torch.device("cpu"),
        env_origins=torch.zeros(2, 3),
        terrain_levels=torch.zeros(2, dtype=torch.long),
        terrain_types=torch.zeros(2, dtype=torch.long),
    )
    apply_terrain_context(
        terrain,
        {
            "terrain_level": torch.tensor([-1, -1]),
            "terrain_type": torch.tensor([-1, -1]),
            "env_origin": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        },
    )

    torch.testing.assert_close(terrain.env_origins, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    torch.testing.assert_close(terrain.terrain_levels, torch.zeros(2, dtype=torch.long))


def test_apply_terrain_context_writes_curriculum_attributes():
    terrain = SimpleNamespace(
        device=torch.device("cpu"),
        terrain_levels=torch.zeros(2, dtype=torch.long),
        terrain_types=torch.zeros(2, dtype=torch.long),
        env_origins=torch.zeros(2, 3),
    )
    apply_terrain_context(
        terrain,
        {
            "terrain_level": torch.tensor([1, 2]),
            "terrain_type": torch.tensor([3, 4]),
            "env_origin": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        },
    )

    torch.testing.assert_close(terrain.terrain_levels, torch.tensor([1, 2]))
    torch.testing.assert_close(terrain.terrain_types, torch.tensor([3, 4]))
    torch.testing.assert_close(terrain.env_origins, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
