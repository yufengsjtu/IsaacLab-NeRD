# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for sharded eager and optimized lazy dataset loading."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
import torch.distributed as dist
from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM
from isaaclab_neural.contacts.tensor_utils import ContactTokenMoments, MaskedContactMoments
from isaaclab_neural.data.datasets import LazyBatchTransitionDataset, LazyTrajectoryDataset, TrajectoryDataset
from isaaclab_neural.train.trainers import _reset_validation_iterators, _synchronize_running_mean_std
from isaaclab_neural.utils.running_mean_std import RunningMeanStd
from torch.utils.data import DataLoader


def _write_token_dataset(path, *, num_trajectories: int = 5, trajectory_length: int = 4) -> None:
    tokens = np.zeros((num_trajectories, trajectory_length, 3, CONTACT_TOKEN_DIM), dtype=np.float32)
    tokens[..., 0] = 1.0
    states = np.zeros((num_trajectories, trajectory_length, 2), dtype=np.float32)
    states[..., 0] = np.arange(num_trajectories, dtype=np.float32).reshape(-1, 1)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        group.attrs["mode"] = "trajectory"
        group.attrs["total_transitions"] = num_trajectories * trajectory_length
        group.attrs["contact_token_frame"] = "world_v1"
        group.attrs["contact_identity_schema"] = "world_owner_v1"
        group.create_dataset("states", data=states)
        group.create_dataset("next_states", data=states)
        group.create_dataset("joint_f", data=np.zeros((num_trajectories, trajectory_length, 1), dtype=np.float32))
        group.create_dataset("contact_tokens", data=tokens)
        group.create_dataset(
            "contact_token_overflow",
            data=np.zeros((num_trajectories, trajectory_length), dtype=np.int64),
        )
        group.create_dataset(
            "contact_token_body_ids",
            data=np.zeros((num_trajectories, trajectory_length, 3), dtype=np.int64),
        )
        group.create_dataset(
            "contact_token_world_ids",
            data=np.zeros((num_trajectories, trajectory_length, 3), dtype=np.int64),
        )
        group.create_dataset("root_body_q", data=np.zeros((num_trajectories, trajectory_length, 7), dtype=np.float32))
        group.create_dataset("gravity_dir", data=np.zeros((num_trajectories, trajectory_length, 3), dtype=np.float32))


def test_eager_trajectory_rank_shards_are_disjoint_and_complete(tmp_path) -> None:
    path = tmp_path / "tokens.hdf5"
    _write_token_dataset(path)

    full = TrajectoryDataset(path, sample_sequence_length=2)
    shard0 = TrajectoryDataset(path, sample_sequence_length=2, rank=0, world_size=2)
    shard1 = TrajectoryDataset(path, sample_sequence_length=2, rank=1, world_size=2)

    assert shard0.global_trajectory_indices.tolist() == [0, 2, 4]
    assert shard1.global_trajectory_indices.tolist() == [1, 3]
    assert set(shard0.global_trajectory_indices).isdisjoint(set(shard1.global_trajectory_indices))
    assert sorted(
        np.concatenate((shard0.global_trajectory_indices, shard1.global_trajectory_indices)).tolist()
    ) == list(range(5))
    assert len(shard0) + len(shard1) == len(full)
    assert shard0[0]["contact_tokens"].shape == (2, 3, CONTACT_TOKEN_DIM)
    assert shard0[0]["contact_token_body_ids"].dtype == torch.long
    assert shard0[0]["contact_token_world_ids"].dtype == torch.long


def test_contact_token_loader_rejects_ambiguous_frame_metadata(tmp_path) -> None:
    path = tmp_path / "legacy_tokens.hdf5"
    _write_token_dataset(path)
    with h5py.File(path, "r+") as handle:
        del handle["data"].attrs["contact_token_frame"]

    with pytest.raises(ValueError, match="contact_token_frame='world_v1'"):
        TrajectoryDataset(path, sample_sequence_length=2)


def test_eager_sharding_applies_global_max_capacity_before_rank_split(tmp_path) -> None:
    path = tmp_path / "tokens.hdf5"
    _write_token_dataset(path)

    shard0 = TrajectoryDataset(path, sample_sequence_length=2, max_capacity=12, rank=0, world_size=2)
    shard1 = TrajectoryDataset(path, sample_sequence_length=2, max_capacity=12, rank=1, world_size=2)

    assert shard0.global_trajectory_indices.tolist() == [0, 2]
    assert shard1.global_trajectory_indices.tolist() == [1]


def test_lazy_trajectory_preserves_tokens_with_persistent_workers(tmp_path) -> None:
    path = tmp_path / "tokens.hdf5"
    _write_token_dataset(path)
    dataset = LazyTrajectoryDataset(path, sample_sequence_length=2)
    loader = DataLoader(dataset, batch_size=2, num_workers=2, persistent_workers=True)
    iterator = None
    try:
        for _ in range(2):
            iterator = iter(loader)
            batch = next(iterator)
            assert batch["contact_tokens"].shape == (2, 2, 3, CONTACT_TOKEN_DIM)
            assert batch["contact_token_overflow"].dtype == torch.long
    finally:
        if iterator is not None:
            iterator._shutdown_workers()
        dataset.close()


def test_lazy_batch_transition_preserves_contact_token_axes(tmp_path) -> None:
    path = tmp_path / "tokens.hdf5"
    _write_token_dataset(path)
    dataset = LazyBatchTransitionDataset(batch_size=2, hdf5_dataset_path=path)
    try:
        sample = dataset[0]
    finally:
        dataset.close()

    assert sample["contact_tokens"].shape == (2, 3, CONTACT_TOKEN_DIM)
    assert sample["contact_token_overflow"].dtype == torch.long


def _mock_all_reduce(monkeypatch, remote_values: list[torch.Tensor]) -> None:
    values = iter(remote_values)

    def fake_all_reduce(tensor: torch.Tensor, op=None) -> None:
        del op
        tensor.add_(next(values).to(device=tensor.device, dtype=tensor.dtype))

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)


def test_running_mean_std_synchronize_matches_global_moments(monkeypatch) -> None:
    rms = RunningMeanStd(shape=(1,), device="cpu")
    rms.update(torch.tensor([[1.0], [3.0]]), batch_dim=True)
    _mock_all_reduce(
        monkeypatch,
        [
            torch.tensor(2.0),
            torch.tensor([12.0]),
            torch.tensor([74.0]),
        ],
    )

    _synchronize_running_mean_std(rms)

    torch.testing.assert_close(rms.mean, torch.tensor([4.0]), atol=1.0e-4, rtol=0.0)
    torch.testing.assert_close(rms.var, torch.tensor([5.0]), atol=5.0e-4, rtol=0.0)
    torch.testing.assert_close(rms.count, torch.tensor(4.0001), atol=1.0e-5, rtol=0.0)


def test_contact_moments_synchronize_before_finalize(monkeypatch) -> None:
    masks = torch.tensor([[[True, False]]])
    values = torch.tensor([[[1.0, 0.0]]])
    moments = MaskedContactMoments.from_batch(values, masks)
    moments.update(values, masks)
    _mock_all_reduce(
        monkeypatch,
        [
            torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            torch.tensor([[0.0], [3.0]], dtype=torch.float64),
            torch.tensor([[0.0], [9.0]], dtype=torch.float64),
        ],
    )

    moments.synchronize()
    rms, counts = moments.finalize("cpu", min_samples=1)

    torch.testing.assert_close(counts, torch.ones(2))
    torch.testing.assert_close(rms.mean, torch.tensor([[1.0], [3.0]]))


def test_contact_token_moments_synchronize(monkeypatch) -> None:
    tokens = torch.zeros(1, 1, 1, CONTACT_TOKEN_DIM)
    tokens[..., 0] = 1.0
    tokens[..., 4] = 2.0
    moments = ContactTokenMoments(CONTACT_TOKEN_DIM, "cpu")
    moments.update(tokens)
    remote_sum = torch.zeros(CONTACT_TOKEN_DIM, dtype=torch.float64)
    remote_square_sum = torch.zeros(CONTACT_TOKEN_DIM, dtype=torch.float64)
    remote_sum[0] = 1.0
    remote_sum[4] = 4.0
    remote_square_sum[0] = 1.0
    remote_square_sum[4] = 16.0
    _mock_all_reduce(monkeypatch, [torch.tensor(1.0), remote_sum, remote_square_sum])

    moments.synchronize()
    rms = moments.finalize("cpu")

    torch.testing.assert_close(rms.mean[4], torch.tensor(3.0))
    torch.testing.assert_close(rms.var[4], torch.tensor(1.0))
    torch.testing.assert_close(rms.count, torch.tensor(2.0))


def test_validation_iterators_are_recreated_each_epoch() -> None:
    loaders = {"valid": DataLoader(torch.arange(8), batch_size=2, shuffle=True)}

    first = _reset_validation_iterators(loaders)
    second = _reset_validation_iterators(loaders)

    assert first["valid"] is not second["valid"]
