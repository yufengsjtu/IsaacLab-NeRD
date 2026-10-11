# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learnability probes preserve diagnostic inputs and report actual PD saturation."""

import json

import h5py
import numpy as np
import pytest
import torch
from isaaclab_neural.eval.learnability import _clone_batch, pd_input_statistics, shifted_saturated_actions


def test_error_groups_partition_generalized_state_without_hiding_hand_error():
    from isaaclab_neural.eval.learnability import g1_error_groups

    names = ["left_hip_pitch_joint", "right_knee_joint", "torso_joint", "left_shoulder_pitch_joint", "left_zero_joint"]
    groups = g1_error_groups(names)
    width = 13 + 2 * len(names)
    assert sorted(i for group in groups.values() for i in group) == list(range(width))
    assert groups["hands"] == [11, 22]
    assert groups["legs"] == [7, 8, 18, 19]
    errors = torch.zeros(2, width)
    errors[:, groups["hands"]] = 3
    contributions = {key: errors[:, ids].sum(-1) / width for key, ids in groups.items()}
    torch.testing.assert_close(sum(contributions.values()), errors.mean(-1))
    assert all(not value.any() for key, value in contributions.items() if key != "hands")


def test_equal_data_sampling_is_unique_stratified_and_does_not_touch_global_rng():
    from types import SimpleNamespace

    from isaaclab_neural.eval.learnability import sample_context_indices

    dataset = SimpleNamespace(
        rows=[(0, i) for i in range(20)] + [(1, i) for i in range(20)] + [(2, i) for i in range(20)]
    )
    state = np.random.get_state()
    chosen = sample_context_indices(dataset, 17, seed=400)
    assert len(set(chosen)) == 17
    assert chosen == sample_context_indices(dataset, 17, seed=400)
    counts = [sum(dataset.rows[i][0] == file_id for i in chosen) for file_id in range(3)]
    assert max(counts) - min(counts) <= 1
    assert np.array_equal(state[1], np.random.get_state()[1])
    with pytest.raises(ValueError):
        sample_context_indices(dataset, 61, seed=400)


def test_old_retention_cannot_be_compensated_by_new_data_gain_or_certified_by_proxy():
    from isaaclab_neural.eval.coverage import retention_adaptation_score

    baseline = {"old": 1.0, "new_a": 10.0, "new_b": 2.0}
    improved = {"old": 0.99, "new_a": 5.0, "new_b": 1.8}
    result = retention_adaptation_score(
        baseline, improved, old_names=["old"], new_names=["new_a", "new_b"], old_kind="original"
    )
    assert result["new_relative_loss"] == pytest.approx(0.7)
    assert result["admissible"] and result["old_preserved"]
    regressed = retention_adaptation_score(
        baseline, {**improved, "old": 1.01}, old_names=["old"], new_names=["new_a", "new_b"], old_kind="original"
    )
    assert not regressed["admissible"] and not regressed["old_preserved"]
    proxy = retention_adaptation_score(
        baseline, improved, old_names=["old"], new_names=["new_a", "new_b"], old_kind="proxy"
    )
    assert proxy["proxy_admissible"] and not proxy["admissible"] and not proxy["original_retention_verified"]
    resampled = retention_adaptation_score(
        baseline, improved, old_names=["old"], new_names=["new_a", "new_b"], old_kind="resampled_recipe"
    )
    assert resampled["resampled_admissible"] and resampled["resampled_retention_verified"]
    assert not resampled["original_retention_verified"] and not resampled["admissible"]
    missing = retention_adaptation_score(
        baseline, improved, old_names=[], new_names=["new_a", "new_b"], old_kind="missing"
    )
    assert missing["old_preserved"] is None and not missing["admissible"]
    zero = retention_adaptation_score(
        {"old": 1.0, "new": 0.0}, {"old": 1.0, "new": 0.0}, old_names=["old"], new_names=["new"], old_kind="original"
    )
    assert zero["new_relative_loss"] == 1 and not zero["admissible"]
    with pytest.raises(ValueError):
        retention_adaptation_score(
            baseline, {**improved, "new_a": float("nan")}, old_names=["old"], new_names=["new_a"], old_kind="original"
        )


def test_new_data_selection_and_audit_never_split_a_world_or_accept_training_worlds():
    from isaaclab_neural.eval.coverage import split_new_context_rows

    identities = np.array(
        [[3, 0, 9, 0], [1, 0, 10, 0], [3, 0, 10, 0], [1, 1, 80, 1], [4, 2, 90, 2], [2, 5, 50, 1]], dtype=np.int64
    )
    selection, audit = split_new_context_rows(identities, selection_worlds=[1, 2], audit_worlds=[3, 4])
    assert selection == [1, 3, 5] and audit == [0, 2, 4]
    with pytest.raises(ValueError):
        split_new_context_rows(identities, selection_worlds=[1, 2, 3], audit_worlds=[3, 4])
    with pytest.raises(ValueError):
        split_new_context_rows(np.vstack((identities, [0, 0, 10, 0])), selection_worlds=[1, 2], audit_worlds=[3, 4])


def test_recipe_window_sampling_is_unique_nonoverlapping_and_uses_private_rng():
    from isaaclab_neural.eval.coverage import sample_recipe_windows

    mapping = np.array([(trajectory, start) for trajectory in range(2) for start in range(22)])
    state = np.random.get_state()
    chosen = sample_recipe_windows(mapping, history=10, count=5, seed=13)
    assert len(chosen) == len(set(chosen)) == 5
    assert chosen == sample_recipe_windows(mapping, history=10, count=5, seed=13)
    assert np.array_equal(state[1], np.random.get_state()[1])
    for trajectory in range(2):
        starts = sorted(mapping[i, 1] for i in chosen if mapping[i, 0] == trajectory)
        assert all(b - a >= 10 for a, b in zip(starts, starts[1:]))
    with pytest.raises(ValueError):
        sample_recipe_windows(mapping, history=10, count=7, seed=13)


def test_recipe_retention_uses_training_mixture_not_equal_source_relative_losses():
    from isaaclab_neural.eval.coverage import recipe_loss

    metrics = {"a": {"teacher_forced_loss": 10.0}, "b": {"teacher_forced_loss": 1.0}}
    assert recipe_loss(metrics, {"a": 0.2, "b": 0.8}) == pytest.approx(2.8)
    assert recipe_loss(metrics, {"a": 2.0, "b": 8.0}) == pytest.approx(2.8)
    with pytest.raises(ValueError):
        recipe_loss(metrics, {"a": 1.0})


def test_recipe_replay_keeps_pool_size_and_only_last_anchor_targets():
    from isaaclab_neural.eval.coverage import merge_recipe_replay, recipe_replay_counts

    weights = {"policy": 0.2, "noise": 0.2, "ckpt500": 1 / 15, "ckpt1000": 1 / 15, "ckpt1499": 1 / 15, "pool": 0.4}
    counts = recipe_replay_counts(weights, 768)
    assert list(counts.values()) == [154, 154, 51, 51, 51, 307]
    assert sum(counts.values()) == 768
    sources = {}
    for name, n in (("a", 2), ("b", 3)):
        target = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3, 1)
        sources[name] = {
            "inputs": {"states": target.clone()},
            "target": target[:, -1].clone(),
            "sequence_target": target,
            "truth": target[:, -1].clone(),
            "families": [name],
            "source_hashes": {name: name},
            "indices": list(range(n)),
            "window_mapping": [[0, i] for i in range(n)],
        }
    combined = merge_recipe_replay(sources, expected_count=5)
    assert combined["target"].shape == (5, 1) and "sequence_target" not in combined
    torch.testing.assert_close(combined["target"], torch.cat([row["target"] for row in sources.values()]))
    assert combined["source_names"] == ["a", "a", "b", "b", "b"]
    full = merge_recipe_replay(sources, expected_count=5, keep_sequence_targets=True)
    torch.testing.assert_close(full["sequence_target"], torch.cat([row["sequence_target"] for row in sources.values()]))
    torch.testing.assert_close(full["target"], combined["target"])
    sources["b"]["target"].zero_()
    with pytest.raises(ValueError):
        merge_recipe_replay(sources, expected_count=5)


def test_recipe_replay_cannot_reuse_evaluation_files_or_seed_groups():
    from isaaclab_neural.eval.coverage import check_recipe_replay_separation

    held = [{"sha256": "held", "spec": {"seed": 4000}}]
    check_recipe_replay_separation([{"sha256": "new", "spec": {"seed": 7000}}], held)
    with pytest.raises(ValueError):
        check_recipe_replay_separation([{"sha256": "held", "spec": {"seed": 7000}}], held)
    with pytest.raises(ValueError):
        check_recipe_replay_separation([{"sha256": "new", "spec": {"seed": 4000}}], held)


def test_target_shift_changes_only_one_already_saturated_coordinate():
    actions = torch.zeros(2, 3)
    pd = torch.tensor([[4.0, 9.0, 0.0], [-8.0, -5.0, 0.0]])
    limits = torch.tensor([3.0, 3.0, 3.0])
    scale = torch.full_like(actions, 0.5)
    changed, joints = shifted_saturated_actions(actions, pd, limits, scale, delta_target=0.25)
    assert joints.tolist() == [0, 1]
    torch.testing.assert_close(changed, torch.tensor([[0.5, 0.0, 0.0], [0.0, -0.5, 0.0]]))
    torch.testing.assert_close(pd.clamp(-limits, limits), (pd + (changed - actions) * scale).clamp(-limits, limits))
    assert not actions.any()
    with pytest.raises(ValueError, match="saturated"):
        shifted_saturated_actions(actions, torch.zeros_like(pd), limits, scale, delta_target=0.25)


def test_cached_batch_is_not_normalized_in_place_twice():
    original = {"joint_f": torch.tensor([1.0, 2.0]), "supervision_mask": torch.tensor([False, True])}
    working = _clone_batch(original)
    working["joint_f"].mul_(10)
    torch.testing.assert_close(original["joint_f"], torch.tensor([1.0, 2.0]))
    assert working["supervision_mask"].dtype == torch.bool


def test_pd_statistics_distinguish_unclipped_and_recorded_effort(tmp_path):
    path = tmp_path / "pd.hdf5"
    states = np.zeros((2, 2, 15), dtype=np.float32)
    targets = np.array([[[1.0], [2.0]], [[0.0], [0.0]]], dtype=np.float32)
    effort = np.zeros((2, 2, 7), dtype=np.float32)
    effort[0, :, 6] = [2.0, 3.0]
    with h5py.File(path, "w") as file:
        file.attrs["opd_metadata"] = json.dumps(
            {
                "source_kind": "mjwarp",
                "protocol": {
                    "stiffness": [2.0],
                    "damping": [0.0],
                    "effort_limits": [3.0],
                    "joint_names": ["test_joint"],
                },
            }
        )
        file.create_dataset("data/states", data=states)
        file.create_dataset("data/action_targets", data=targets)
        file.create_dataset("data/joint_f", data=effort)
        file.create_dataset("contacts/native_pairs/frame_offsets", data=[0, 0, 1, 1, 1])
        file.create_dataset("opd/query_identity", data=[[0, 0, 1, 0], [1, 0, 2, 0]])
    result = pd_input_statistics(path)
    assert result["pd_formula_vs_recorded_abs_max_nm"] == 0
    assert result["anchor_any_saturation_fraction"] == 0.5
    assert result["contact_count_change_fraction"] == 0.5
