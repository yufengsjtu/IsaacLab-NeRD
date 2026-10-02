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
from isaaclab_neural.train import trainers as trainers_module
from isaaclab_neural.train.trainers import VanillaTrainer
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
    loader = DataLoader(dataset, batch_size=2, num_workers=2, persistent_workers=True, multiprocessing_context="spawn")
    iterator = None
    try:
        for _ in range(2):
            iterator = iter(loader)
            batch = next(iterator)
            assert batch["contact_tokens"].shape == (2, 2, 3, CONTACT_TOKEN_DIM)
            assert batch["contact_token_overflow"].dtype == torch.long
    finally:
        if iterator is not None:
            getattr(iterator, "_shutdown_workers")()
        dataset.close()


@pytest.mark.parametrize("sampling", ["canonical", "sliding"])
@pytest.mark.parametrize("rank,world_size", [(0, 1), (1, 2)])
def test_unique_batch_reads_preserve_order_duplicates_masks_and_fields(tmp_path, sampling, rank, world_size):
    from isaaclab_neural.data.datasets import create_trajectory_dataset

    path = tmp_path / "tokens.hdf5"
    _write_token_dataset(path, trajectory_length=8)
    with h5py.File(path, "a") as file:
        file["data"].attrs["recording_id"] = "batched-token-fixture"
        file.create_dataset("data/terminated", data=np.zeros((5, 8), dtype=bool))
        file.create_dataset("data/truncated", data=np.zeros((5, 8), dtype=bool))
        file.create_dataset("context/trajectories/source_env_id", data=np.arange(5, dtype=np.int64))
        file.create_dataset("context/trajectories/source_start_frame", data=np.zeros(5, dtype=np.int64))
    kwargs = dict(
        load_mode="lazy",
        hdf5_dataset_path=path,
        sample_sequence_length=3,
        max_capacity=29,
        capacity_unit="unique_transition",
        window_sampling=sampling,
        rank=rank,
        world_size=world_size,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
    )
    full = create_trajectory_dataset(**kwargs)
    selected = create_trajectory_dataset(**kwargs, read_keys=("states", "next_states"))
    try:
        indices = [2, 0, 1, 1, len(full) - 1, 0] * 13
        expected = [full[i] for i in indices]
        batched = selected.__getitems__(indices)
        for actual, reference in zip(batched, expected, strict=True):
            assert not {"joint_f", "state_world_id", "source_env_id"} & actual.keys()
            for name in actual:
                torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
        from isaaclab_neural.data.datasets import WeightedMixedTrajectoryDataset

        second = create_trajectory_dataset(**kwargs, read_keys=("states", "next_states"))
        second.supervision_source_index = 1
        mixed = WeightedMixedTrajectoryDataset({"first": selected, "second": second}, {"first": 2.0, "second": 1.0})
        try:
            mixed_indices = [len(mixed) - 1, 0, int(mixed.source_boundaries[0]), 2, 0]
            for actual, reference in zip(
                mixed.__getitems__(mixed_indices), [mixed[i] for i in mixed_indices], strict=True
            ):
                for name in reference:
                    torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
        finally:
            second.close()
        assert selected.__getitems__([]) == []
        with pytest.raises(IndexError):
            selected.__getitems__([len(selected)])
        loader = DataLoader(selected, batch_size=3, num_workers=1, multiprocessing_context="spawn")
        first = next(iter(loader))
        torch.testing.assert_close(first["states"], torch.stack([selected[i]["states"] for i in range(3)]))
        with pytest.raises(ValueError, match="read keys"):
            create_trajectory_dataset(**kwargs, read_keys=("missing",))
    finally:
        full.close()
        selected.close()


@pytest.mark.parametrize("sampling", ["canonical", "sliding"])
@pytest.mark.parametrize("rank,world_size", [(0, 1), (1, 2)])
def test_eager_preload_is_bounded_and_matches_lazy(tmp_path, monkeypatch, sampling, rank, world_size):
    from isaaclab_neural.data import datasets

    path = tmp_path / "bounded.hdf5"
    _write_token_dataset(path, trajectory_length=8)
    with h5py.File(path, "a") as file:
        file["data"].attrs["recording_id"] = "bounded-eager-fixture"
        file.create_dataset("data/terminated", data=np.zeros((5, 8), dtype=bool))
        file.create_dataset("data/truncated", data=np.zeros((5, 8), dtype=bool))
        file.create_dataset("context/trajectories/source_env_id", data=np.arange(5, dtype=np.int64))
        file.create_dataset("context/trajectories/source_start_frame", data=np.zeros(5, dtype=np.int64))
    kwargs = dict(
        hdf5_dataset_path=path,
        sample_sequence_length=3,
        max_capacity=29,
        capacity_unit="unique_transition",
        window_sampling=sampling,
        rank=rank,
        world_size=world_size,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
        read_keys=("states", "next_states", "source_env_id"),
    )
    lazy = datasets.create_trajectory_dataset(load_mode="lazy", **kwargs)
    original = datasets.read_and_project_frames
    projected_frames = []

    def project(group, indices, projection):
        projected_frames.append(indices.copy())
        return original(group, indices, projection)

    monkeypatch.setattr(datasets, "read_and_project_frames", project)
    eager = datasets.create_trajectory_dataset(load_mode="eager", eager_preload_frames=8, **kwargs)
    try:
        assert projected_frames and max(len(indices) for indices in projected_frames) <= 8
        np.testing.assert_array_equal(
            np.concatenate(projected_frames), (eager.global_trajectory_indices[:, None] * 8 + np.arange(8)).reshape(-1)
        )
        assert eager.capacity_stats == lazy.capacity_stats
        for i in range(len(eager)):
            expected, actual = lazy[i], eager[i]
            assert expected.keys() == actual.keys()
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        with pytest.raises(ValueError, match="eager_preload_frames"):
            datasets.create_trajectory_dataset(load_mode="eager", eager_preload_frames=0, **kwargs)
    finally:
        lazy.close()
        eager.close()


def test_eager_random_projection_preserves_whole_pool_rng(tmp_path):
    from isaaclab_neural.data.datasets import create_trajectory_dataset

    path = tmp_path / "random.hdf5"
    _write_token_dataset(path, trajectory_length=8)
    with h5py.File(path, "a") as file:
        file["data"].attrs["recording_id"] = "random-eager-fixture"
        file.create_dataset("data/terminated", data=np.zeros((5, 8), dtype=bool))
        file.create_dataset("data/truncated", data=np.zeros((5, 8), dtype=bool))
        file.create_dataset("context/trajectories/source_env_id", data=np.arange(5, dtype=np.int64))
        file.create_dataset("context/trajectories/source_start_frame", data=np.zeros(5, dtype=np.int64))
    kwargs = dict(
        load_mode="eager",
        hdf5_dataset_path=path,
        sample_sequence_length=3,
        max_capacity=29,
        capacity_unit="unique_transition",
        window_sampling="sliding",
        contact_representation="flat",
        num_contacts_per_env=2,
        contact_packing_policy="random",
    )
    torch.manual_seed(17)
    reference = create_trajectory_dataset(eager_preload_frames=4096, **kwargs)
    rng = torch.get_rng_state()
    torch.manual_seed(17)
    actual = create_trajectory_dataset(eager_preload_frames=8, **kwargs)
    try:
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key in reference.dataset:
            np.testing.assert_array_equal(actual.dataset[key], reference.dataset[key])
    finally:
        reference.close()
        actual.close()


def test_hyperslab_window_reads_preserve_overlaps_order_and_shapes(tmp_path):
    from isaaclab_neural.data.datasets import _LazyHdf5Accessor

    path = tmp_path / "windows.hdf5"
    _write_token_dataset(path, trajectory_length=8)
    with h5py.File(path, "a") as file:
        file.create_dataset("data/primary_body_q", data=np.arange(5 * 8 * 3 * 7, dtype=np.float32).reshape(5, 8, 3, 7))
        file.create_dataset("data/reward", data=np.arange(40, dtype=np.float64).reshape(5, 8) / 3)
    accessor = _LazyHdf5Accessor(path)
    windows = [(3, 2), (1, 1), (1, 0), (1, 1), (3, 5)]
    keys = ["states", "primary_body_q", "reward"]
    try:
        actual = accessor.read_windows(keys, windows, 3)
        for sample, (row, start) in zip(actual, windows, strict=True):
            for key in keys:
                expected = np.asarray(accessor.dataset(key)[row, start : start + 3]).astype(np.float32)
                np.testing.assert_array_equal(sample[key], expected)
        assert accessor.read_windows(keys, [], 3) == []
        with pytest.raises(IndexError):
            accessor.read_windows(keys, [(0, 7)], 3)
    finally:
        accessor.close()


def test_lazy_accessor_caches_handles_and_drops_them_on_pickle(tmp_path):
    import pickle

    from isaaclab_neural.data.datasets import _LazyHdf5Accessor

    path = tmp_path / "tokens.hdf5"
    _write_token_dataset(path)
    accessor = _LazyHdf5Accessor(path)
    original = accessor.dataset("states")
    assert accessor.dataset("states") is original
    restored = pickle.loads(pickle.dumps(accessor))
    try:
        np.testing.assert_array_equal(restored.dataset("states")[:], original[:])
        accessor.close()
        assert not original.id.valid
        np.testing.assert_array_equal(restored.dataset("states")[:], accessor.dataset("states")[:])
    finally:
        accessor.close()
        restored.close()


def _write_overlapping_dataset(path, world_count=2):
    worlds = torch.arange(world_count).repeat_interleave(2)
    starts = torch.tensor([0, 32]).repeat(world_count)
    ids = worlds[:, None] * 1000 + starts[:, None] + torch.arange(64)
    states = torch.stack((ids.float(), ids.float() + 0.5), dim=-1)
    write_rollouts_to_hdf5(
        path,
        {
            "states": states,
            "next_states": states + 1,
            "joint_f": torch.zeros(2 * world_count, 64, 1),
            "root_body_q": torch.zeros(2 * world_count, 64, 7),
            "root_body_qd": torch.zeros(2 * world_count, 64, 6),
            "gravity_dir": torch.zeros(2 * world_count, 64, 3),
            "terminated": torch.zeros(2 * world_count, 64, dtype=torch.bool),
            "truncated": torch.zeros(2 * world_count, 64, dtype=torch.bool),
            "contacts": {},
            "trajectory_context": {"source_env_id": worlds, "source_start_frame": starts},
        },
        env_name="Test-Env",
    )
    with h5py.File(path, "a") as handle:
        handle["data"].attrs["recording_id"] = "overlap-fixture"
        handle.create_dataset("replay/frames/states", shape=(world_count, 96, 2), dtype="f4")


@pytest.mark.parametrize("capacity", [1, 9, 10, 63, 64, 65, 191, 192])
def test_sliding_unique_capacity_preserves_overlapping_supervision(tmp_path, capacity):
    from isaaclab_neural.data.datasets import create_trajectory_dataset

    path = tmp_path / "overlap.hdf5"
    _write_overlapping_dataset(path)
    modes = []
    for mode in ("lazy", "eager"):
        dataset = create_trajectory_dataset(
            load_mode=mode,
            hdf5_dataset_path=path,
            max_capacity=capacity,
            sample_sequence_length=10,
            capacity_unit="unique_transition",
            window_sampling="sliding",
        )
        windows, targets, identities = [], [], set()
        for i in range(len(dataset)):
            sample = dataset[i]
            mask = sample["supervision_mask"]
            windows.append(tuple(sample["states"][:, 0].tolist()))
            targets.extend(sample["states"][mask, 0].tolist())
            identities.update(sample["supervision_target_ids"][mask].tolist())
        assert len(windows) == len(set(windows))
        assert len(set(targets)) == dataset.capacity_stats["unique_selected"] == capacity
        assert identities == set(range(capacity))
        if capacity == 192:
            assert len(windows) == 174 and len(targets) == 1740
            assert targets.count(32.0) == 10
        if capacity == 64:
            assert len(windows) == 55 and len(targets) == 550
        if capacity == 65:
            assert len(windows) == 56 and len(targets) == 560
        modes.append((windows, targets))
        dataset.close()
    assert modes[0] == modes[1]


def test_sliding_index_matches_bruteforce_for_overlaps_gaps_and_boundaries(tmp_path):
    from isaaclab_neural.data.transition_index import build_transition_index

    rng = np.random.default_rng(11)
    for case in range(12):
        starts = np.sort(rng.integers(0, 220, size=12))
        lengths = rng.integers(10, 65, size=12)
        pool = sorted(
            {
                frame
                for start, length in zip(starts, lengths, strict=True)
                for frame in range(int(start), int(start + length))
            }
        )
        capacity = int(rng.integers(1, len(pool) + 1))
        selected = set(pool[:capacity])
        path = tmp_path / f"random-{case}.hdf5"
        with h5py.File(path, "w") as file:
            file.create_dataset("data/states", shape=(12, 64, 2), dtype="f4")
            file["data"].attrs["recording_id"] = f"random-{case}"
            file.create_dataset("data/traj_lengths", data=lengths)
            file.create_dataset("data/terminated", data=np.zeros((12, 64), dtype=bool))
            file.create_dataset("data/truncated", data=np.zeros((12, 64), dtype=bool))
            file.create_dataset("context/trajectories/source_env_id", data=np.zeros(12, dtype=np.int64))
            file.create_dataset("context/trajectories/source_start_frame", data=starts)
        index = build_transition_index(path, max_capacity=capacity, sequence_length=10, window_sampling="sliding")
        expected = {
            start
            for begin, length in zip(starts, lengths, strict=True)
            for start in range(int(begin), int(begin + length - 9))
            if set(range(start, start + 10)) <= selected
        }
        full, observed, identities = set(), set(), {}
        assert index.sliding_blocks is not None
        for row, first, count, supervised, target, _ in index.sliding_blocks:
            for offset in range(int(count)):
                source_start = int(starts[row] + first + offset)
                if supervised == 10:
                    assert source_start not in full
                    full.add(source_start)
                for step in range(int(supervised)):
                    observed.add(source_start + step)
                    identities[source_start + step] = int(target + offset + step)
        assert full == expected and observed == selected
        assert identities == {frame: index for index, frame in enumerate(pool[:capacity])}


@pytest.mark.parametrize("mode", ["lazy", "eager"])
def test_sliding_ddp_keeps_source_rows_and_targets_on_one_rank(tmp_path, mode):
    from isaaclab_neural.data.datasets import create_trajectory_dataset

    path = tmp_path / "overlap.hdf5"
    _write_overlapping_dataset(path, world_count=8)
    rows, targets, fingerprints = set(), set(), set()
    for rank in range(8):
        dataset = create_trajectory_dataset(
            load_mode=mode,
            hdf5_dataset_path=path,
            max_capacity=768,
            sample_sequence_length=10,
            capacity_unit="unique_transition",
            window_sampling="sliding",
            rank=rank,
            world_size=8,
        )
        local_rows = set(dataset.global_trajectory_indices.tolist())
        local_targets = set()
        for i in range(len(dataset)):
            sample = dataset[i]
            local_targets.update(sample["supervision_target_ids"][sample["supervision_mask"]].tolist())
        assert not rows.intersection(local_rows) and not targets.intersection(local_targets)
        rows.update(local_rows)
        targets.update(local_targets)
        fingerprints.add(dataset.capacity_stats["selection_sha256"])
        assert dataset.capacity_stats["rank_unique_selected"] == 96
        dataset.close()
    assert rows == set(range(16)) and targets == set(range(768)) and len(fingerprints) == 1


def test_sliding_seen_counts_source_targets_not_window_exposures(tmp_path):
    from isaaclab_neural.data.datasets import create_trajectory_dataset

    path = tmp_path / "overlap.hdf5"
    _write_overlapping_dataset(path)
    dataset = create_trajectory_dataset(
        load_mode="lazy",
        hdf5_dataset_path=path,
        max_capacity=65,
        sample_sequence_length=10,
        capacity_unit="unique_transition",
        window_sampling="sliding",
    )
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.cfg = {"dataset_capacity_contract": {"train": {"policy": dataset.capacity_stats}}}
    for _ in range(2):
        for batch in DataLoader(dataset, batch_size=7):
            trainer.record_unique_training_targets(batch)
    assert trainer.unique_training_targets_seen == 65
    dataset.close()


@pytest.mark.parametrize("capacity", [1, 63, 65, 191, 192, 1000])
def test_unique_capacity_is_exact_and_eager_lazy_targets_match(tmp_path, capacity):
    from isaaclab_neural.data.datasets import create_trajectory_dataset

    path = tmp_path / "overlap.hdf5"
    _write_overlapping_dataset(path)
    targets = []
    for mode in ("lazy", "eager"):
        dataset = create_trajectory_dataset(
            load_mode=mode,
            hdf5_dataset_path=path,
            sample_sequence_length=10,
            max_capacity=capacity,
            capacity_unit="unique_transition",
        )
        seen = []
        for index in range(len(dataset)):
            sample = dataset[index]
            assert sample["states"].shape == (10, 2)
            mask = sample["supervision_mask"]
            assert mask.dtype == torch.bool and bool(mask.any())
            seen.extend(sample["states"][mask, 0].tolist())
        assert len(seen) == len(set(seen)) == min(capacity, 192)
        assert dataset.capacity_stats["unique_selected"] == min(capacity, 192)
        assert dataset.capacity_stats["unique_available"] == 192
        targets.append(seen)
        dataset.close()
    assert targets[0] == targets[1]


@pytest.mark.parametrize("world_size", [1, 2, 8])
@pytest.mark.parametrize("mode", ["lazy", "eager"])
def test_unique_targets_partition_globally_before_ddp(tmp_path, world_size, mode):
    from isaaclab_neural.data.datasets import create_trajectory_dataset

    path = tmp_path / "overlap.hdf5"
    _write_overlapping_dataset(path)
    seen = set()
    fingerprints = set()
    for rank in range(world_size):
        dataset = create_trajectory_dataset(
            load_mode=mode,
            hdf5_dataset_path=path,
            sample_sequence_length=10,
            max_capacity=191,
            capacity_unit="unique_transition",
            rank=rank,
            world_size=world_size,
        )
        local = set()
        for index in range(len(dataset)):
            sample = dataset[index]
            local.update(sample["states"][sample["supervision_mask"], 0].tolist())
        assert not seen.intersection(local)
        seen.update(local)
        assert dataset.capacity_stats["unique_selected"] == 191
        assert len(local) == dataset.capacity_stats["rank_supervised_targets"]
        fingerprints.add(dataset.capacity_stats["selection_sha256"])
        dataset.close()
    assert len(seen) == 191 and len(fingerprints) == 1


def test_unique_capacity_gate_matches_loader_instead_of_raw_clip_limit(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "osmo_scripts"))
    from lib.keyboard_data import validate_dataset_scale

    path = tmp_path / "overlap.hdf5"
    _write_overlapping_dataset(path)
    experiment = {
        "minimum_unique_transitions": {"train": 192},
        "dataset_specs": [{"filename": path.name, "split": "train", "num_transitions": 192}],
    }
    with pytest.raises(ValueError, match="160.*192"):
        validate_dataset_scale(tmp_path, experiment, max_capacity=192)
    experiment.update(dataset_capacity_unit="unique_transition", training_sequence_length=10)
    assert validate_dataset_scale(tmp_path, experiment, max_capacity=192) == {"train": 192}
    with pytest.raises(ValueError, match="191.*192"):
        validate_dataset_scale(tmp_path, experiment, max_capacity=191)


def test_unique_index_requires_identity_and_rejects_reset_crossing(tmp_path):
    from isaaclab_neural.data.transition_index import build_transition_index

    path = tmp_path / "overlap.hdf5"
    _write_overlapping_dataset(path)
    with h5py.File(path, "a") as file:
        file["data/terminated"][0, 10] = True
    with pytest.raises(ValueError, match="reset boundary"):
        build_transition_index(path, max_capacity=192, sequence_length=10)
    with h5py.File(path, "a") as file:
        file["data/terminated"][0, 10] = False
        del file["context/trajectories/source_start_frame"]
    with pytest.raises(ValueError, match="identities"):
        build_transition_index(path, max_capacity=192)


def test_unique_view_deduplicates_recording_aliases(tmp_path):
    import json
    import shutil

    from isaaclab_neural.data.transition_index import build_transition_index

    first, second, view = (tmp_path / name for name in ("a.hdf5", "b.hdf5", "view.hdf5"))
    _write_overlapping_dataset(first)
    shutil.copy2(first, second)
    layout = h5py.VirtualLayout(shape=(8, 64, 2), dtype="f4")
    layout[:4] = h5py.VirtualSource(str(first), "data/states", shape=(4, 64, 2))
    layout[4:] = h5py.VirtualSource(str(second), "data/states", shape=(4, 64, 2))
    with h5py.File(view, "w") as file:
        file.create_virtual_dataset("data/states", layout)
        file["data"].attrs["keyboard_contract"] = json.dumps({"training_parts": [first.name, second.name]})
    index = build_transition_index(view, max_capacity=1000, sequence_length=10)
    assert index.unique_available == index.unique_selected == 192
    assert index.raw_frame_slots == 512
    assert len(index.recording_ids) == 1
    with h5py.File(second, "a") as file:
        file["data/states"][0, 0, 0] = -1
    with pytest.raises(ValueError, match="identity collision"):
        build_transition_index(view, max_capacity=1000, sequence_length=10)


def test_unique_mixture_enforces_global_budget_and_disjoint_recordings(tmp_path):
    from isaaclab_neural.data.datasets import create_weighted_trajectory_dataset

    first, second = tmp_path / "a.hdf5", tmp_path / "b.hdf5"
    _write_overlapping_dataset(first)
    _write_overlapping_dataset(second)
    sources = {
        "a": {"path": first, "weight": 1.0, "max_capacity": 25},
        "b": {"path": second, "weight": 1.0, "max_capacity": 25},
    }
    with pytest.raises(ValueError, match="global training capacity"):
        create_weighted_trajectory_dataset(
            train_datasets=sources, load_mode="lazy", capacity_unit="unique_transition", max_capacity=49
        )
    with pytest.raises(ValueError, match="share recording identities"):
        create_weighted_trajectory_dataset(
            train_datasets=sources, load_mode="lazy", capacity_unit="unique_transition", max_capacity=50
        )
    with h5py.File(second, "a") as file:
        file["data"].attrs["recording_id"] = "second-recording"
    mixed = create_weighted_trajectory_dataset(
        train_datasets=sources, load_mode="lazy", capacity_unit="unique_transition", max_capacity=50
    )
    assert sum(dataset.capacity_stats["unique_selected"] for dataset in mixed.datasets.values()) == 50
    assert sum(int(mixed[index]["supervision_mask"].sum()) for index in range(len(mixed))) == 50
    mixed.close()


def test_unique_seen_does_not_count_repeated_optimizer_samples(tmp_path):
    from isaaclab_neural.data.datasets import create_trajectory_dataset

    path = tmp_path / "overlap.hdf5"
    _write_overlapping_dataset(path)
    dataset = create_trajectory_dataset(
        load_mode="lazy",
        hdf5_dataset_path=path,
        sample_sequence_length=10,
        max_capacity=65,
        capacity_unit="unique_transition",
    )
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    for _ in range(2):
        for batch in DataLoader(dataset, batch_size=3):
            trainer.record_unique_training_targets(batch)
    assert trainer.unique_training_targets_seen == 65
    dataset.close()


@pytest.mark.parametrize("sampling", ["canonical", "sliding"])
def test_statistics_follow_the_selected_window_objective(tmp_path, sampling):
    from types import SimpleNamespace

    from isaaclab_neural.data.datasets import create_trajectory_dataset

    path = tmp_path / "overlap.hdf5"
    _write_overlapping_dataset(path)
    dataset = create_trajectory_dataset(
        load_mode="lazy",
        hdf5_dataset_path=path,
        sample_sequence_length=10,
        max_capacity=65,
        capacity_unit="unique_transition",
        window_sampling=sampling,
    )
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.device = "cpu"
    trainer.dataset_window_sampling = sampling
    trainer.batch_size = 2
    trainer.dataset_capacity_unit = "unique_transition"
    trainer.is_distributed = False
    trainer.is_main_process = True
    trainer.neural_env = SimpleNamespace(frame_dt=0.01)
    trainer.neural_solver = SimpleNamespace(
        contact_mode="newton_native",
        convert_next_states_to_prediction=lambda **kwargs: kwargs["next_states"] - kwargs["states"],
    )
    trainer._make_dataloader = lambda **kwargs: DataLoader(kwargs["dataset"], batch_size=3)
    trainer.preprocess_data_batch = lambda batch: {
        key: value for key, value in batch.items() if key in ("states", "next_states", "supervision_mask")
    }
    trainer.compute_dataset_statistics(dataset)
    reference = (
        torch.cat([torch.arange(start, start + 10) for start in range(56)]).float()
        if sampling == "sliding"
        else torch.arange(65).float()
    )
    assert float(trainer.dataset_rms["states"].count) == pytest.approx(len(reference) + 0.0001)
    assert float(trainer.dataset_rms["states"].mean[0]) == pytest.approx(float(reference.mean()), abs=1e-3)
    assert float(trainer.dataset_rms["states"].var[0]) == pytest.approx(float(reference.var(unbiased=False)), abs=5e-3)
    dataset.close()


def _unique_ddp_gradient_worker(rank, rendezvous):
    from isaaclab_neural.train.trainers import distributed_supervision_loss
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        model = DistributedDataParallel(model)
        count = 1 if rank == 0 else 3
        target = 1.0 if rank == 0 else 3.0
        loss = (model(torch.ones(count, 1)) - target).square().mean()
        distributed_supervision_loss(loss, torch.tensor(float(count))).backward()
        torch.testing.assert_close(model.module.weight.grad, torch.tensor([[-5.0]]))
        trainer = _metric_trainer()
        trainer.is_distributed, trainer.world_size = True, 2
        batches = [{"supervision_mask": torch.ones(1, 10, dtype=torch.bool), "maximum": float(rank + 1)}]
        _, metrics, _ = trainer.one_epoch(False, batches, iter(batches), 1)
        assert metrics["key_q_m_max"] == 2.0
    finally:
        dist.destroy_process_group()


def test_unique_ddp_loss_matches_global_target_mean(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_unique_ddp_gradient_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)


def _sliding_statistics_worker(rank, path, rendezvous):
    from types import SimpleNamespace

    from isaaclab_neural.data.datasets import create_trajectory_dataset

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    dataset = None
    try:
        dataset = create_trajectory_dataset(
            load_mode="lazy",
            hdf5_dataset_path=path,
            max_capacity=191,
            sample_sequence_length=10,
            capacity_unit="unique_transition",
            window_sampling="sliding",
            rank=rank,
            world_size=2,
        )
        trainer = VanillaTrainer.__new__(VanillaTrainer)
        trainer.device = "cpu"
        trainer.batch_size = 16
        trainer.dataset_capacity_unit = "unique_transition"
        trainer.dataset_window_sampling = "sliding"
        trainer.is_distributed = True
        trainer.train_dataset_rank_sharded = True
        trainer.is_main_process = False
        trainer.neural_env = SimpleNamespace(frame_dt=0.01)
        trainer.neural_solver = SimpleNamespace(
            contact_mode="newton_native",
            convert_next_states_to_prediction=lambda **kwargs: kwargs["next_states"] - kwargs["states"],
        )
        trainer._make_dataloader = lambda **kwargs: DataLoader(
            kwargs["dataset"], batch_size=16, num_workers=1, multiprocessing_context="spawn"
        )
        trainer.preprocess_data_batch = lambda batch: {
            key: value for key, value in batch.items() if key in ("states", "next_states", "supervision_mask")
        }
        trainer.compute_dataset_statistics(dataset)
        expected = torch.cat(
            [torch.arange(start, start + 10) for start in range(87)]
            + [torch.arange(start + 1000, start + 1010) for start in range(86)]
        ).float()
        assert float(trainer.dataset_rms["states"].count) == pytest.approx(len(expected), abs=0.01)
        assert float(trainer.dataset_rms["states"].mean[0]) == pytest.approx(float(expected.mean()), abs=0.01)
        trainer.cfg = {"dataset_capacity_contract": {"train": {"policy": dataset.capacity_stats}}}
        for batch in DataLoader(dataset, batch_size=16):
            trainer.record_unique_training_targets(batch)
        total = torch.tensor(trainer.unique_training_targets_seen)
        dist.all_reduce(total)
        assert int(total) == 191
    finally:
        if dataset is not None:
            dataset.close()
        dist.destroy_process_group()


def test_sliding_ddp_window_statistics_and_unique_seen(tmp_path):
    import torch.multiprocessing as mp

    path = tmp_path / "overlap.hdf5"
    _write_overlapping_dataset(path)
    mp.spawn(_sliding_statistics_worker, args=(str(path), f"file://{tmp_path / 'rendezvous'}"), nprocs=2, join=True)


def _population_weighted_gradient_worker(rank, rendezvous):
    from isaaclab_neural.train.trainers import distributed_supervision_loss
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1.0)
        model = DistributedDataParallel(model)
        trainer = VanillaTrainer.__new__(VanillaTrainer)
        trainer.is_distributed = trainer.train_dataset_rank_sharded = True
        trainer.dataset_capacity_unit = "unique_transition"
        trainer.dataset_window_sampling = "sliding"
        trainer.train_dataset = [None] * (1 if rank == 0 else 3)
        data = {"supervision_mask": torch.ones(1, 1, dtype=torch.bool)}
        scale = trainer._batch_sampling_scale(data, True)
        loss = model(torch.tensor([[1.0 if rank == 0 else 3.0]])).square().mean()
        distributed_supervision_loss(loss, data["supervision_mask"].sum() * scale).backward()
        assert model.module.weight.grad.item() == pytest.approx(14.0)
        metrics_trainer = _metric_trainer()
        metrics_trainer.is_distributed = True
        metrics_trainer.world_size = 2

        def metric_loss(data, train):
            metrics_trainer._metric_statistics = {
                "robot_normal_contact_precision": (float(rank), 1.0 if rank == 0 else 99.0),
                "robot_normal_active_RMSE_N": (100.0 * rank, float(rank)),
            }
            return torch.tensor(1.0), {
                "robot_normal_RMSE_N": 10.0 * rank,
                "robot_normal_contact_precision": 0.0,
                "robot_normal_active_RMSE_N": 10.0 * rank,
            }

        metrics_trainer.compute_loss = metric_loss
        _, metrics, _ = metrics_trainer.one_epoch(False, [data], iter([data]), 1)
        assert metrics["robot_normal_RMSE_N"] == pytest.approx(50.0**0.5)
        assert metrics["robot_normal_contact_precision"] == pytest.approx(0.01)
        assert metrics["robot_normal_active_RMSE_N"] == pytest.approx(10.0)
    finally:
        dist.destroy_process_group()


def test_ddp_unequal_populations_match_global_window_objective(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_population_weighted_gradient_worker, args=(f"file://{tmp_path / 'population'}",), nprocs=2, join=True)


def test_training_worker_import_has_no_distributed_side_effects(monkeypatch):
    import runpy
    import sys
    from unittest.mock import Mock

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(sys, "argv", ["train-worker"])
    initialize = Mock()
    select_device = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", select_device)
    monkeypatch.setattr(dist, "init_process_group", initialize)
    runpy.run_module("isaaclab_neural.train.train", run_name="__mp_main__")
    initialize.assert_not_called()
    select_device.assert_not_called()
    assert sys.argv == ["train-worker"]


@pytest.mark.parametrize("recomputed", [False, True])
def test_resume_preserves_explicitly_recomputed_statistics(recomputed):
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.device = "cpu"
    trainer.dataset_statistics_recomputed = recomputed
    trainer.loss_weights = torch.tensor([2.0])
    trainer.contact_rms_counts = {"contact_tokens": torch.tensor(9.0)}
    trainer._checkpoint = {
        "loss_weights": torch.tensor([1.0]),
        "contact_rms_counts": {"contact_tokens": torch.tensor(3.0)},
    }
    trainer._restore_training_state()
    torch.testing.assert_close(trainer.loss_weights, torch.tensor([2.0 if recomputed else 1.0]))
    assert float(trainer.contact_rms_counts["contact_tokens"]) == (9.0 if recomputed else 3.0)


def _metric_trainer():
    from isaaclab_neural.utils.time_report import TimeReport

    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.neural_model = torch.nn.Identity()
    trainer.device = "cpu"
    trainer.is_main_process = trainer.is_distributed = False
    trainer.world_size = 1
    trainer.time_report = TimeReport()
    trainer.time_report.add_timers(["dataloader", "compute_loss", "backward", "other"])
    trainer.preprocess_data_batch = lambda data: data
    trainer.compute_loss = lambda data, train: (torch.tensor(data["maximum"]), {"key_q_m_max": data["maximum"]})
    return trainer


def test_epoch_maximum_is_not_averaged_over_supervised_targets():
    trainer = _metric_trainer()
    batches = [
        {"supervision_mask": torch.arange(10)[None] < count, "maximum": maximum}
        for count, maximum in ((1, 0.01), (9, 0.0))
    ]
    loss, metrics, _ = trainer.one_epoch(False, batches, iter(batches), 2)
    assert loss == pytest.approx(0.001)
    assert metrics["key_q_m_max"] == pytest.approx(0.01)


def test_epoch_rmse_and_contact_ratios_use_global_sufficient_statistics():
    trainer = _metric_trainer()
    batches = [{"supervision_mask": torch.ones(1, 1, dtype=torch.bool), "index": index} for index in range(2)]

    def loss(data, train):
        index = data["index"]
        trainer._metric_statistics = {
            "robot_normal_contact_precision": (1.0, 1.0 if index == 0 else 100.0),
            "robot_normal_active_RMSE_N": (0.0 if index == 0 else 100.0, float(index)),
        }
        return torch.tensor(1.0), {
            "robot_normal_RMSE_N": 10.0 * index,
            "robot_normal_contact_precision": 1.0 if index == 0 else 0.01,
            "robot_normal_active_RMSE_N": 10.0 * index,
        }

    trainer.compute_loss = loss
    _, metrics, _ = trainer.one_epoch(False, batches, iter(batches), 2)
    assert metrics["robot_normal_RMSE_N"] == pytest.approx(50.0**0.5)
    assert metrics["robot_normal_contact_precision"] == pytest.approx(2.0 / 101.0)
    assert metrics["robot_normal_active_RMSE_N"] == pytest.approx(10.0)


class _DataLoaderTrainerHarness(VanillaTrainer):
    make_dataloader = VanillaTrainer._make_dataloader


@pytest.mark.parametrize("workers", [0, 1])
def test_trainer_dataloader_avoids_fork_and_preserves_batches(workers) -> None:
    trainer = _DataLoaderTrainerHarness.__new__(_DataLoaderTrainerHarness)
    trainer.num_data_workers = workers
    trainer.persistent_workers = False
    trainer.pin_memory = False
    trainer.prefetch_factor = 2
    trainer.collate_fn = None
    loader = trainer.make_dataloader(
        dataset=torch.utils.data.TensorDataset(torch.arange(6)),
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )
    if workers:
        assert loader.multiprocessing_context is not None
        assert loader.multiprocessing_context.get_start_method() == "spawn"
    else:
        assert loader.multiprocessing_context is None
    for _ in range(2):
        assert torch.equal(torch.cat([batch[0] for batch in loader]), torch.arange(6))


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

    getattr(trainers_module, "_synchronize_running_mean_std")(rms)

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
    loaders = {"valid": DataLoader(torch.utils.data.TensorDataset(torch.arange(8)), batch_size=2, shuffle=True)}

    reset_iterators = getattr(trainers_module, "_reset_validation_iterators")
    first = reset_iterators(loaders)
    second = reset_iterators(loaders)

    assert first["valid"] is not second["valid"]


def _write_random_native_dataset(path, *, precision: str, num_trajectories: int = 5, horizon: int = 8) -> None:
    """Variable pair counts, mixed signs of separation and two-sided bodies, for projection parity."""
    generator = torch.Generator().manual_seed(7)
    counts = torch.randint(0, 5, (num_trajectories, horizon), generator=generator)
    pair_count = int(counts.sum())
    native = {name: torch.zeros(pair_count, dtype=torch.int64) for name in NATIVE_CONTACT_INTEGER_FIELDS}
    native.update({name: torch.randn((pair_count, 3), generator=generator) for name in NATIVE_CONTACT_VECTOR_FIELDS})
    native.update({name: torch.randn(pair_count, generator=generator) for name in NATIVE_CONTACT_SCALAR_FIELDS})
    native["frame_counts"] = counts
    worlds = torch.arange(num_trajectories).repeat_interleave(counts.sum(dim=1))
    native["world_ids"] = worlds
    native["body_slots_0"] = torch.randint(0, 3, (pair_count,), generator=generator)
    native["body_ids_0"] = worlds * 3 + native["body_slots_0"]
    static = torch.rand(pair_count, generator=generator) < 0.5
    native["body_slots_1"] = torch.where(static, -1, torch.randint(0, 3, (pair_count,), generator=generator))
    native["body_ids_1"] = torch.where(static, -1, worlds * 3 + native["body_slots_1"])
    states = torch.randn((num_trajectories, horizon, 2), generator=generator)
    write_rollouts_to_hdf5(
        path,
        {
            "states": states,
            "next_states": states + 1,
            "joint_f": torch.randn((num_trajectories, horizon, 1), generator=generator),
            "root_body_q": torch.zeros((num_trajectories, horizon, 7)),
            "root_body_qd": torch.zeros((num_trajectories, horizon, 6)),
            "gravity_dir": torch.zeros((num_trajectories, horizon, 3)),
            "terminated": torch.zeros((num_trajectories, horizon), dtype=torch.bool),
            "truncated": torch.zeros((num_trajectories, horizon), dtype=torch.bool),
            "contacts": {},
            "native_contacts": native,
            "trajectory_context": {
                "source_env_id": torch.arange(num_trajectories),
                "source_start_frame": torch.zeros(num_trajectories, dtype=torch.int64),
                **{
                    name: torch.arange(num_trajectories)
                    for name in ("state_world_id", "root_world_id", "contact_world_id")
                },
            },
        },
        env_name="Test-Env",
        native_contact_precision=precision,
    )
    with h5py.File(path, "a") as file:
        file["data"].attrs["recording_id"] = "gpu-resident-fixture"


@pytest.mark.parametrize("sampling", ["canonical", "sliding"])
@pytest.mark.parametrize("rank,world_size", [(0, 1), (1, 2)])
@pytest.mark.parametrize("precision", ["full", "compact_v1"])
@pytest.mark.parametrize("contact_filter", ["none", "solver_active"])
def test_gpu_resident_batches_match_eager_windows(tmp_path, sampling, rank, world_size, precision, contact_filter):
    from isaaclab_neural.data.datasets import WeightedMixedTrajectoryDataset, create_trajectory_dataset
    from isaaclab_neural.data.gpu_resident import GpuBatchLoader, GpuResidentTrajectoryDataset
    from isaaclab_neural.data.transition_index import SUPERVISION_WINDOW

    path = tmp_path / "native.hdf5"
    _write_random_native_dataset(path, precision=precision)
    kwargs = dict(
        hdf5_dataset_path=path,
        sample_sequence_length=3,
        max_capacity=29,
        capacity_unit="unique_transition",
        window_sampling=sampling,
        rank=rank,
        world_size=world_size,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
        contact_filter=contact_filter,
        read_keys=("states", "next_states", "joint_f"),
        eager_preload_frames=16,
    )
    eager = create_trajectory_dataset(load_mode="eager", **kwargs)
    resident_kwargs = {key: value for key, value in kwargs.items() if key not in ("hdf5_dataset_path", "capacity_unit")}
    # Compact resident pairs are exact only for values already stored at compact precision.
    gpu = GpuResidentTrajectoryDataset(path, device="cpu", compact_pairs=precision == "compact_v1", **resident_kwargs)
    half = create_trajectory_dataset(load_mode="gpu", device="cpu", float16_keys=("states",), **kwargs)
    try:
        assert len(gpu) == len(eager) and gpu.capacity_stats == eager.capacity_stats
        indices = [len(eager) - 1, 0, 2, 2, 1, *range(len(eager))]
        batch = gpu.get_batch(indices)
        for position, index in enumerate(indices):
            reference = eager[index]
            assert reference.keys() == batch.keys()
            for key, value in reference.items():
                torch.testing.assert_close(batch[key][position], value, rtol=0, atol=0)
        assert bool((batch["contact_tokens"][..., 0] > 0.5).any())
        torch.testing.assert_close(half.get_batch(indices)["states"], batch["states"], rtol=1e-3, atol=1e-3)

        loader = GpuBatchLoader(gpu, batch_size=4, shuffle=True, seed=3)
        windows = torch.cat([chunk[SUPERVISION_WINDOW] for chunk in loader])
        assert len(loader) == -(-len(gpu) // 4)
        everything = gpu.get_batch(list(range(len(gpu))))
        torch.testing.assert_close(windows.sort().values, everything[SUPERVISION_WINDOW].sort().values)
        second = GpuResidentTrajectoryDataset(
            path, device="cpu", compact_pairs=precision == "compact_v1", **resident_kwargs
        )
        second.supervision_source_index = 1
        mixed = WeightedMixedTrajectoryDataset({"first": gpu, "second": second}, {"first": 2.0, "second": 1.0})
        mixed_indices = [len(mixed) - 1, 0, int(mixed.source_boundaries[0]), 2]
        mixed_batch = next(iter(GpuBatchLoader(mixed, batch_size=len(mixed), shuffle=False)))
        for index in mixed_indices:
            for key, value in mixed[index].items():
                torch.testing.assert_close(mixed_batch[key][index], value, rtol=0, atol=0)
    finally:
        eager.close()
        gpu.close()
        half.close()
