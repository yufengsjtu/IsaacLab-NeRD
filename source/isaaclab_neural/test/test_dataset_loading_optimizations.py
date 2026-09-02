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
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_TOKEN_DIM,
    NATIVE_CONTACT_INTEGER_FIELDS,
    NATIVE_CONTACT_SCALAR_FIELDS,
    NATIVE_CONTACT_VECTOR_FIELDS,
)
from isaaclab_neural.contacts.tensor_utils import ContactTokenMoments, MaskedContactMoments
from isaaclab_neural.data.datasets import LazyBatchTransitionDataset, LazyTrajectoryDataset, TrajectoryDataset
from isaaclab_neural.data.hdf5 import write_rollouts_to_hdf5
from isaaclab_neural.train.trainers import _reset_validation_iterators, _synchronize_running_mean_std
from isaaclab_neural.utils.running_mean_std import RunningMeanStd
from torch.utils.data import DataLoader


def _write_token_dataset(path, *, num_trajectories: int = 5, trajectory_length: int = 4) -> None:
    pairs_per_frame = 3
    pair_count = num_trajectories * trajectory_length * pairs_per_frame
    native_contacts = {name: torch.zeros(pair_count, dtype=torch.int64) for name in NATIVE_CONTACT_INTEGER_FIELDS}
    native_contacts.update(
        {name: torch.zeros((pair_count, 3), dtype=torch.float32) for name in NATIVE_CONTACT_VECTOR_FIELDS}
    )
    native_contacts.update(
        {name: torch.zeros(pair_count, dtype=torch.float32) for name in NATIVE_CONTACT_SCALAR_FIELDS}
    )
    native_contacts["frame_counts"] = torch.full(
        (num_trajectories, trajectory_length),
        pairs_per_frame,
        dtype=torch.int64,
    )
    native_contacts["body_ids_1"].fill_(-1)
    native_contacts["body_slots_1"].fill_(-1)
    native_contacts["world_ids"] = torch.arange(num_trajectories, dtype=torch.int64).repeat_interleave(
        trajectory_length * pairs_per_frame
    )
    native_contacts["normal_w"][:, 2] = 1.0
    native_contacts["separation"] = -torch.arange(1, pair_count + 1, dtype=torch.float32)
    native_contacts["points_1_w"][:, 2] = native_contacts["separation"]

    states = torch.zeros((num_trajectories, trajectory_length, 2), dtype=torch.float32)
    states[..., 0] = torch.arange(num_trajectories, dtype=torch.float32).reshape(-1, 1)
    world_ids = torch.arange(num_trajectories, dtype=torch.int64)
    write_rollouts_to_hdf5(
        path,
        {
            "states": states,
            "next_states": states.clone(),
            "joint_f": torch.zeros((num_trajectories, trajectory_length, 1), dtype=torch.float32),
            "root_body_q": torch.zeros((num_trajectories, trajectory_length, 7), dtype=torch.float32),
            "root_body_qd": torch.zeros((num_trajectories, trajectory_length, 6), dtype=torch.float32),
            "gravity_dir": torch.zeros((num_trajectories, trajectory_length, 3), dtype=torch.float32),
            "contacts": {},
            "native_contacts": native_contacts,
            "trajectory_context": {
                "state_world_id": world_ids,
                "root_world_id": world_ids,
                "contact_world_id": world_ids,
            },
        },
        env_name="Test-Env",
    )


def test_eager_trajectory_rank_shards_are_disjoint_and_complete(tmp_path) -> None:
    path = tmp_path / "tokens.hdf5"
    _write_token_dataset(path)

    full = TrajectoryDataset(
        path,
        sample_sequence_length=2,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
        contact_packing_policy="body_round_robin_pair_atomic",
    )
    shard0 = TrajectoryDataset(
        path,
        sample_sequence_length=2,
        rank=0,
        world_size=2,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
        contact_packing_policy="body_round_robin_pair_atomic",
    )
    shard1 = TrajectoryDataset(
        path,
        sample_sequence_length=2,
        rank=1,
        world_size=2,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
        contact_packing_policy="body_round_robin_pair_atomic",
    )

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


@pytest.mark.parametrize("metadata_value", [None, "wrong_schema"])
def test_contact_token_loader_rejects_invalid_native_schema_metadata(tmp_path, metadata_value) -> None:
    path = tmp_path / "native_contacts.hdf5"
    _write_token_dataset(path)
    with h5py.File(path, "r+") as handle:
        if metadata_value is None:
            del handle["data"].attrs["contact_storage_schema"]
        else:
            handle["data"].attrs["contact_storage_schema"] = metadata_value

    with pytest.raises(ValueError, match="native_pairs schema v3"):
        TrajectoryDataset(
            path,
            sample_sequence_length=2,
            contact_representation="contact_tokens",
            max_contact_tokens=3,
            contact_packing_policy="body_round_robin_pair_atomic",
        )


def test_eager_sharding_applies_global_max_capacity_before_rank_split(tmp_path) -> None:
    path = tmp_path / "tokens.hdf5"
    _write_token_dataset(path)

    full = TrajectoryDataset(
        path,
        sample_sequence_length=2,
        max_capacity=20,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
        contact_packing_policy="body_round_robin_pair_atomic",
    )
    limited = TrajectoryDataset(
        path,
        sample_sequence_length=2,
        max_capacity=12,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
        contact_packing_policy="body_round_robin_pair_atomic",
    )
    shard0 = TrajectoryDataset(
        path,
        sample_sequence_length=2,
        max_capacity=12,
        rank=0,
        world_size=2,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
        contact_packing_policy="body_round_robin_pair_atomic",
    )
    shard1 = TrajectoryDataset(
        path,
        sample_sequence_length=2,
        max_capacity=12,
        rank=1,
        world_size=2,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
        contact_packing_policy="body_round_robin_pair_atomic",
    )

    assert full.global_trajectory_indices.tolist() == [0, 1, 2, 3, 4]
    assert limited.global_trajectory_indices.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(limited.dataset["states"][:, 0, 0], np.array([0.0, 1.0, 2.0]))
    assert shard0.global_trajectory_indices.tolist() == [0, 2]
    assert shard1.global_trajectory_indices.tolist() == [1]


def test_lazy_trajectory_preserves_tokens_with_persistent_workers(tmp_path) -> None:
    path = tmp_path / "tokens.hdf5"
    _write_token_dataset(path)
    dataset = LazyTrajectoryDataset(
        path,
        sample_sequence_length=2,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
        contact_packing_policy="body_round_robin_pair_atomic",
    )
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
    dataset = LazyBatchTransitionDataset(
        batch_size=2,
        hdf5_dataset_path=path,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
        contact_packing_policy="body_round_robin_pair_atomic",
    )
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
