# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native contact contexts retain short histories and cannot masquerade as trajectories."""

import h5py
import numpy as np
import pytest
import torch
from isaaclab_neural.data.datasets import TrajectoryDataset
from isaaclab_neural.data.opd import ContextBatchSampler, ContextDataset, audit_dataset, write_context_shard
from native_hdf5_samples import make_native_rollouts


def _metadata(split="train"):
    return {
        "physics_dt": 0.005,
        "split": split,
        "policy_hash": "random-policy",
        "dynamics_hash": "fixed-model",
        "source_kind": "nerd",
        "families": ["run:0:0", "run:1:0"],
        "label_origin": "mjwarp",
    }


def test_context_roundtrip_mask_and_legacy_rejection(tmp_path):
    path = tmp_path / "context.hdf5"
    data = make_native_rollouts()
    target = data["next_states"][:, -1].clone()
    write_context_shard(path, data, metadata=_metadata(), labels=target)
    dataset = ContextDataset([path], split="train", physics_dt=0.005, max_history=10)
    sample = dataset[0]
    assert sample["states"].shape == (3, 4)
    assert sample["contact_tokens"].shape == (3, 64, 17)
    assert sample["supervision_mask"].tolist() == [False, False, True]
    torch.testing.assert_close(sample["next_states"][-1], target[0])
    with pytest.raises(ValueError, match="trajectory"):
        TrajectoryDataset(hdf5_dataset_path=str(path), sample_sequence_length=3)
    with pytest.raises(FileExistsError):
        write_context_shard(path, data, metadata=_metadata(), labels=target)
    with pytest.raises(ValueError, match="split"):
        ContextDataset([path], split="validation", physics_dt=0.005)
    with pytest.raises(ValueError, match="physics_dt"):
        ContextDataset([path], split="train", physics_dt=0.02)


def test_full_real_history_roundtrip_and_anchor_compatibility(tmp_path):
    data = make_native_rollouts(num_steps=10)
    states = torch.arange(data["states"].numel(), dtype=torch.float32).reshape_as(data["states"])
    data["states"] = states
    targets = torch.cat((states[:, 1:], states[:, -1:] + 1), dim=1)
    identities = torch.tensor([[0, 2, 29, 0], [1, 3, 39, 0]])
    frames = identities[:, 2, None] + torch.arange(-9, 1)
    episodes = identities[:, 1, None].expand_as(frames)
    metadata = {**_metadata(), "source_kind": "mjwarp", "sequence_label_origin": "observed_mj_post_step"}
    path = tmp_path / "full.hdf5"
    diagnostics = {"query_identity": identities, "history_frame_ids": frames, "history_episode_ids": episodes}
    write_context_shard(
        path, data, metadata=metadata, labels=targets[:, -1], sequence_labels=targets, diagnostics=diagnostics
    )
    dataset = ContextDataset([path], split="train", physics_dt=0.005, supervision="all_positions")
    sample = dataset[0]
    assert sample["supervision_mask"].all()
    torch.testing.assert_close(sample["next_states"], targets[0], rtol=0, atol=0)
    legacy = ContextDataset([path], split="train", physics_dt=0.005)[0]
    assert legacy["supervision_mask"].sum() == 1
    torch.testing.assert_close(legacy["next_states"][-1], targets[0, -1], rtol=0, atol=0)
    with pytest.raises(ValueError, match="real MJ"):
        write_context_shard(
            tmp_path / "student.hdf5",
            data,
            metadata={**metadata, "source_kind": "nerd"},
            labels=targets[:, -1],
            sequence_labels=targets,
            diagnostics=diagnostics,
        )
    bad = frames.clone()
    bad[0, 3] -= 1
    with pytest.raises(ValueError, match="consecutive"):
        write_context_shard(
            tmp_path / "gap.hdf5",
            data,
            metadata=metadata,
            labels=targets[:, -1],
            sequence_labels=targets,
            diagnostics={**diagnostics, "history_frame_ids": bad},
        )
    bad_episodes = episodes.clone()
    bad_episodes[0, 3] += 1
    with pytest.raises(ValueError, match="episode"):
        write_context_shard(
            tmp_path / "reset.hdf5",
            data,
            metadata=metadata,
            labels=targets[:, -1],
            sequence_labels=targets,
            diagnostics={**diagnostics, "history_episode_ids": bad_episodes},
        )
    with pytest.raises(ValueError, match="continuity"):
        write_context_shard(
            tmp_path / "placeholder.hdf5",
            data,
            metadata=metadata,
            labels=targets[:, -1],
            sequence_labels=torch.cat((states[:, :-1], targets[:, -1:]), 1),
            diagnostics=diagnostics,
        )


def test_anchor_only_contexts_cannot_silently_become_full_labels(tmp_path):
    data = make_native_rollouts(num_steps=10)
    path = tmp_path / "legacy.hdf5"
    write_context_shard(
        path, data, metadata={**_metadata(), "source_kind": "mjwarp"}, labels=data["next_states"][:, -1]
    )
    with pytest.raises(ValueError, match="full-history"):
        ContextDataset([path], split="train", physics_dt=0.005, supervision="all_positions")


def test_unlabelled_queries_cannot_train_and_nan_oracle_labels_rejected(tmp_path):
    data = make_native_rollouts()
    path = tmp_path / "queries.hdf5"
    write_context_shard(path, data, metadata=_metadata())
    with pytest.raises(ValueError, match="unlabelled"):
        ContextDataset([path], split="train", physics_dt=0.005)
    with pytest.raises(ValueError, match="finite"):
        write_context_shard(tmp_path / "bad.hdf5", data, metadata=_metadata(), labels=torch.full((2, 4), float("nan")))


def test_bucket_batches_preserve_history_lengths_and_global_rng(tmp_path):
    paths = []
    for length in [1, 3, 10]:
        path = tmp_path / f"h{length}.hdf5"
        data = make_native_rollouts(num_steps=length)
        write_context_shard(path, data, metadata=_metadata(), labels=data["next_states"][:, -1])
        paths.append(path)
    dataset = ContextDataset(paths, split="train", physics_dt=0.005)
    rng = np.random.get_state()
    batches = list(ContextBatchSampler(dataset, batch_size=2, shuffle=True, seed=4))
    assert len(batches) == 3
    for batch in batches:
        assert len({dataset.lengths[i] for i in batch}) == 1
    np.testing.assert_equal(np.random.get_state(), rng)
    assert {len(dataset[batch[0]]["states"]) for batch in batches} == {1, 3, 10}


def test_audit_does_not_treat_zero_force_or_old_contacts_as_trusted(tmp_path):
    path = tmp_path / "old.hdf5"
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.create_dataset("states", data=np.zeros((1, 4, 4), dtype=np.float32))
        data.create_dataset("joint_f", data=np.zeros((1, 4, 2), dtype=np.float32))
        data.create_dataset("contact_tokens", data=np.zeros((1, 4, 64, 17), dtype=np.float32))
    report = audit_dataset(path)
    assert not report["eligible_for_rehearsal"]
    assert report["sampled_joint_f_abs_max"] == 0
    assert report["problems"]


def _sampling_dataset():
    dataset = ContextDataset.__new__(ContextDataset)
    dataset.sources = [source for source in ("reference", "current_mj", "student") for _ in range(4)]
    dataset.lengths = [1, 1, 10, 10] * 3
    dataset.rows = [(0, i) for i in range(12)]
    return dataset


def test_update_sampler_has_exact_source_quotas_and_reproducible_exposure():
    from collections import Counter

    from isaaclab_neural.data.opd import ContextUpdateSampler

    data = _sampling_dataset()
    quotas = {"reference": 8, "current_mj": 4, "student": 4}
    state = np.random.get_state()
    a = ContextUpdateSampler(data, 16, source_counts=quotas, length_weights={1: 0.2, 10: 0.8}, seed=11)
    b = ContextUpdateSampler(data, 16, source_counts=quotas, length_weights={1: 0.2, 10: 0.8}, seed=11)
    for _ in range(20):
        batch_a, batch_b = next(iter(a)), next(iter(b))
        assert batch_a == batch_b
        assert Counter(data.sources[i] for i in batch_a) == quotas
    np.testing.assert_equal(np.random.get_state(), state)
    assert a.audit.summary()["anchor_exposures"] == 320
    assert a.audit.summary()["index_trace_sha256"] == b.audit.summary()["index_trace_sha256"]
    with pytest.raises(ValueError, match="sum"):
        ContextUpdateSampler(data, 32, source_counts=quotas, length_weights={1: 1}, seed=0)
    with pytest.raises(ValueError, match="bucket"):
        ContextUpdateSampler(data, 16, source_counts=quotas, length_weights={2: 1}, seed=0)


def test_update_collation_never_pads_and_keeps_every_anchor():
    from isaaclab_neural.data.opd import collate_context_update

    samples = []
    for length in [1, 10, 1, 3]:
        mask = torch.zeros(length, dtype=torch.bool)
        mask[-1] = True
        samples.append({"states": torch.ones(length, 4), "supervision_mask": mask})
    batches = collate_context_update(samples)
    assert sorted((b["states"].shape[0], b["states"].shape[1]) for b in batches) == [(1, 3), (1, 10), (2, 1)]
    assert sum(int(b["supervision_mask"].sum()) for b in batches) == 4


@pytest.mark.parametrize("count", [256, 257, 511, 512, 1000, 1024])
def test_current_epochs_cover_without_replacement_and_avoid_small_tails(count):
    from isaaclab_neural.data.opd import CurrentEpochSampler

    state = np.random.get_state()
    sampler = CurrentEpochSampler(count, reference_count=17, replay_stage_sizes=[11, 19], seed=7)
    other = CurrentEpochSampler(count, reference_count=17, replay_stage_sizes=[11, 19], seed=7)
    for _ in range(2):
        batches = list(sampler)
        assert batches == list(other)
        assert sorted(i for batch in batches for i in batch["current"]) == list(range(count))
        assert len(batches) == count // 256
        for batch in batches:
            assert 256 <= len(batch["current"]) < 512
            assert len(batch["reference"]) + len(batch["replay"]) == len(batch["current"])
            assert batch["source_weights"] == {"current": 0.5, "reference": 0.25, "replay": 0.25}
    np.testing.assert_equal(state, np.random.get_state())


def test_current_epoch_empty_replay_and_insufficient_data():
    from isaaclab_neural.data.opd import CurrentEpochSampler

    batches = list(CurrentEpochSampler(1024, reference_count=17, replay_stage_sizes=[], seed=7))
    assert all(len(b["current"]) == len(b["reference"]) == 256 and not b["replay"] for b in batches)
    assert all(b["source_weights"] == {"current": 0.5, "reference": 0.5, "replay": 0.0} for b in batches)
    with pytest.raises(ValueError, match="current"):
        CurrentEpochSampler(255, reference_count=17, replay_stage_sizes=[], seed=7)
    with pytest.raises(ValueError, match="reference"):
        CurrentEpochSampler(256, reference_count=0, replay_stage_sizes=[], seed=7)
