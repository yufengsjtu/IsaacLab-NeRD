# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for contact reconstruction diagnostic edge cases."""

import math

import h5py
import numpy as np
import pytest
import torch
from isaaclab_neural.eval.contact_reconstruction_diagnostic import (
    aggregate_contact_metrics,
    aggregate_contact_token_metrics,
    analyze_dataset_token_row_ownership,
    contact_metrics,
    contact_token_metrics,
    input_difference_metrics,
    load_random_dataset_batch,
    raw_token_frame_metrics,
    root_replay_metrics,
    validate_contact_metrics,
    validate_contact_token_metrics,
)


def _inputs(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size, num_contacts = mask.shape
    return {
        "states": torch.zeros(batch_size, 1, 4),
        "root_body_q": torch.zeros(batch_size, 1, 7),
        "joint_f": torch.zeros(batch_size, 1, 2),
        "gravity_dir": torch.zeros(batch_size, 1, 3),
        "contact_masks": mask.unsqueeze(1),
        "contact_points_0": torch.zeros(batch_size, 1, num_contacts * 3),
        "contact_points_1": torch.zeros(batch_size, 1, num_contacts * 3),
        "contact_normals": torch.zeros(batch_size, 1, num_contacts * 3),
        "contact_depths": torch.zeros(batch_size, 1, num_contacts),
        "contact_thicknesses_0": torch.zeros(batch_size, 1, num_contacts),
        "contact_thicknesses_1": torch.zeros(batch_size, 1, num_contacts),
    }


def _token_inputs(tokens: torch.Tensor, overflow: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if overflow is None:
        overflow = torch.zeros(tokens.shape[0], 1, dtype=torch.long)
    return {
        "contact_tokens": tokens.unsqueeze(1),
        "contact_token_overflow": overflow,
    }


def test_root_replay_metrics_report_maximal_state_errors():
    dataset = {
        "states": torch.zeros(2, 1, 4),
        "root_body_q": torch.zeros(2, 1, 7),
        "root_body_qd": torch.zeros(2, 1, 6),
    }
    runtime = {key: value.clone() for key, value in dataset.items()}
    runtime["root_body_q"][0, 0, 0] = 3.0
    runtime["root_body_qd"][1, 0, 2] = 2.0

    metrics = root_replay_metrics(dataset, runtime)

    assert metrics["root_position_l2_max"] == pytest.approx(3.0)
    assert metrics["root_linear_velocity_l2_max"] == pytest.approx(2.0)


def test_raw_token_frame_metrics_remove_common_root_translation():
    dataset_tokens = torch.zeros(1, 1, 17)
    dataset_tokens[..., 0] = 1.0
    dataset = {
        "contact_tokens": dataset_tokens.unsqueeze(1),
        "root_body_q": torch.zeros(1, 1, 7),
    }
    runtime = {key: value.clone() for key, value in dataset.items()}
    translation = torch.tensor([4.0, -2.0, 1.0])
    runtime["root_body_q"][0, 0, :3] = translation
    runtime["contact_tokens"][0, 0, 0, 4:7] = translation

    metrics = raw_token_frame_metrics(dataset, runtime)

    assert metrics["world_point_delta_l2_max"] == pytest.approx(float(torch.linalg.vector_norm(translation)))
    assert metrics["point_minus_root_delta_l2_max"] == pytest.approx(0.0)


def test_analyze_dataset_token_row_ownership_reports_swapped_rows(tmp_path, capsys):
    path = tmp_path / "tokens.hdf5"
    tokens = np.zeros((2, 1, 1, 17), dtype=np.float32)
    tokens[..., 0] = 1.0
    with h5py.File(path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        data_group.attrs["contact_token_frame"] = "world_v1"
        data_group.attrs["contact_identity_schema"] = "world_owner_v1"
        data_group.create_dataset("contact_tokens", data=tokens)
        data_group.create_dataset("contact_token_body_ids", data=np.asarray([[[4]], [[70]]], dtype=np.int64))
        data_group.create_dataset("contact_token_world_ids", data=np.asarray([[[1]], [[0]]], dtype=np.int64))
        trajectory_group = dataset_file.create_group("context").create_group("trajectories")
        trajectory_group.create_dataset("state_world_id", data=np.asarray([0, 1], dtype=np.int64))
        trajectory_group.create_dataset("root_world_id", data=np.asarray([0, 1], dtype=np.int64))
        trajectory_group.create_dataset("contact_world_id", data=np.asarray([0, 1], dtype=np.int64))

    with pytest.raises(ValueError, match="ownership schema mismatch"):
        analyze_dataset_token_row_ownership(str(path), torch.tensor([0, 0, 0, 0, 1, 1, 1, 0]))

    output = capsys.readouterr().out
    assert "recorded_token_world_row_mismatch_count=2" in output
    assert "owner_body_runtime_unavailable_count=1" in output
    assert "missing_valid_owner_body_count=0" in output
    assert "state_contact_world_mismatch_count=0" in output


def test_contact_token_metrics_report_geometry_and_validity_mismatch():
    dataset_tokens = torch.zeros(2, 3, 17)
    runtime_tokens = dataset_tokens.clone()
    dataset_tokens[0, 0, 0] = 1.0
    runtime_tokens[0, 0, 0] = 1.0
    runtime_tokens[0, 0, 4] = 0.25
    runtime_tokens[1, 1, 0] = 1.0

    metrics = contact_token_metrics(_token_inputs(dataset_tokens), _token_inputs(runtime_tokens))

    assert metrics["valid_mismatch_count"] == 1.0
    assert metrics["mismatched_env_count"] == 2.0
    assert metrics["point_l2_max"] == 0.25
    with pytest.raises(ValueError, match="Contact-token reconstruction mismatch"):
        validate_contact_token_metrics(metrics, point_tolerance=1.0e-4)


def test_contact_token_metrics_do_not_shift_after_missing_identity_match():
    dataset_tokens = torch.zeros(1, 3, 17)
    runtime_tokens = torch.zeros_like(dataset_tokens)
    dataset_tokens[..., 0] = 1.0
    dataset_tokens[0, :, 4] = torch.tensor([0.0, 1.0, 2.0])
    runtime_tokens[0, :2, 0] = 1.0
    runtime_tokens[0, :2, 4] = torch.tensor([0.0, 2.0])

    metrics = contact_token_metrics(_token_inputs(dataset_tokens), _token_inputs(runtime_tokens))

    assert metrics["valid_mismatch_count"] == 1.0
    assert metrics["categorical_mismatch_count"] == 0.0
    assert metrics["shared_valid_count"] == 2.0
    assert metrics["point_l2_max"] == 0.0


def test_contact_token_metrics_do_not_pair_different_identities():
    dataset_tokens = torch.zeros(1, 1, 17)
    runtime_tokens = torch.zeros_like(dataset_tokens)
    dataset_tokens[..., 0] = 1.0
    runtime_tokens[..., 0] = 1.0
    runtime_tokens[..., 1] = 1.0

    metrics = contact_token_metrics(_token_inputs(dataset_tokens), _token_inputs(runtime_tokens))

    assert metrics["valid_mismatch_count"] == 0.0
    assert metrics["categorical_mismatch_count"] == 2.0
    assert metrics["shared_valid_count"] == 0.0
    assert metrics["point_l2_max"] == 0.0


def test_aggregate_contact_token_metrics_weights_shared_contacts():
    first = contact_token_metrics(
        _token_inputs(torch.tensor([[[1.0] + [0.0] * 16]])),
        _token_inputs(torch.tensor([[[1.0] + [0.0] * 3 + [1.0] + [0.0] * 12]])),
    )
    second_tokens = torch.zeros(1, 3, 17)
    second_tokens[..., 0] = 1.0
    second_runtime = second_tokens.clone()
    second_runtime[..., 4] = 3.0
    second = contact_token_metrics(_token_inputs(second_tokens), _token_inputs(second_runtime))

    summary = aggregate_contact_token_metrics([first, second], batch_size=1)

    assert summary["shared_valid_count"] == 4.0
    assert summary["point_l2_mean"] == 2.5
    assert summary["point_l2_max"] == 3.0


def test_contact_token_metrics_separate_relative_velocity():
    dataset_tokens = torch.zeros(1, 1, 17)
    runtime_tokens = dataset_tokens.clone()
    dataset_tokens[..., 0] = 1.0
    runtime_tokens[..., 0] = 1.0
    runtime_tokens[..., 14] = 4.0

    metrics = contact_token_metrics(_token_inputs(dataset_tokens), _token_inputs(runtime_tokens))

    assert metrics["point_l2_max"] == 0.0
    assert metrics["relative_velocity_l2_max"] == 4.0
    assert metrics["mismatched_env_count"] == 1.0


def test_input_difference_metrics_handles_no_shared_contacts():
    dataset_inputs = _inputs(torch.tensor([[True, False]]))
    runtime_inputs = _inputs(torch.tensor([[False, True]]))

    metrics = input_difference_metrics(dataset_inputs, runtime_inputs, 2)

    assert math.isnan(metrics["contact_points_0_l2_mean_active"])
    assert math.isnan(metrics["contact_points_0_l2_max_active"])
    assert math.isnan(metrics["contact_depths_abs_mean_active"])
    assert math.isnan(metrics["contact_depths_abs_max_active"])


def test_contact_metrics_reports_one_sided_empty_contacts():
    dataset_inputs = _inputs(torch.tensor([[True, False], [True, False]]))
    runtime_inputs = _inputs(torch.tensor([[False, False], [True, False]]))

    metrics = contact_metrics(dataset_inputs, runtime_inputs, 2)

    assert math.isinf(metrics["set_chamfer_distance_mean"])
    assert metrics["mask_mismatch_count"] == 1.0
    assert metrics["mask_total_count"] == 4.0
    assert metrics["mismatched_env_count"] == 1.0
    assert metrics["one_sided_empty_count"] == 1.0
    assert metrics["one_sided_empty_fraction"] == 0.5


def test_contact_metrics_tolerates_slot_permutation():
    mask = torch.tensor([[True, True, False]])
    dataset_inputs = _inputs(mask)
    runtime_inputs = _inputs(mask)
    dataset_points = dataset_inputs["contact_points_1"].reshape(1, 1, 3, 3)
    runtime_points = runtime_inputs["contact_points_1"].reshape(1, 1, 3, 3)
    dataset_points[0, 0, 0, 0] = 1.0
    runtime_points[0, 0, 1, 0] = 1.0

    metrics = contact_metrics(dataset_inputs, runtime_inputs, 3)

    assert metrics["slot_point_distance_max"] == 1.0
    assert metrics["set_hausdorff_distance_max"] == 0.0
    validate_contact_metrics(metrics, tolerance=1.0e-4)


def test_contact_metrics_tolerates_duplicate_contact_within_tolerance():
    dataset_inputs = _inputs(torch.tensor([[True, False]]))
    runtime_inputs = _inputs(torch.tensor([[True, True]]))
    runtime_inputs["contact_points_0"][0, 0, 3] = 5.0e-5
    runtime_inputs["contact_points_1"][0, 0, 3] = 5.0e-5

    metrics = contact_metrics(dataset_inputs, runtime_inputs, 2, deduplication_tolerance=1.0e-4)

    assert metrics["mask_mismatch_count"] == 1.0
    assert metrics["unique_count_abs_diff_max"] == 0.0
    assert metrics["set_hausdorff_distance_max"] == pytest.approx(5.0e-5)
    validate_contact_metrics(metrics, tolerance=1.0e-4)


def test_contact_deduplication_is_permutation_invariant():
    mask = torch.tensor([[True, True, True]])
    dataset_inputs = _inputs(mask)
    runtime_inputs = _inputs(mask)
    dataset_points = torch.zeros(1, 1, 3, 3)
    dataset_points[0, 0, :, 0] = torch.tensor([0.0, 0.09, 0.18])
    runtime_points = dataset_points[:, :, [1, 0, 2]].clone()
    for key in ("contact_points_0", "contact_points_1"):
        dataset_inputs[key].copy_(dataset_points.reshape(1, 1, -1))
        runtime_inputs[key].copy_(runtime_points.reshape(1, 1, -1))

    metrics = contact_metrics(dataset_inputs, runtime_inputs, 3, deduplication_tolerance=0.1)

    assert metrics["dataset_unique_active_mean"] == 1.0
    assert metrics["runtime_unique_active_mean"] == 1.0
    assert metrics["unique_count_abs_diff_max"] == 0.0


def test_contact_matching_resolves_coincident_normal_ties():
    mask = torch.tensor([[True, True]])
    dataset_inputs = _inputs(mask)
    runtime_inputs = _inputs(mask)
    dataset_normals = dataset_inputs["contact_normals"].reshape(1, 1, 2, 3)
    runtime_normals = runtime_inputs["contact_normals"].reshape(1, 1, 2, 3)
    dataset_normals[0, 0, 0, 0] = 1.0
    dataset_normals[0, 0, 1, 1] = 1.0
    runtime_normals[0, 0, 0, 1] = 1.0
    runtime_normals[0, 0, 1, 0] = 1.0

    metrics = contact_metrics(dataset_inputs, runtime_inputs, 2)

    assert metrics["matched_normal_l2_max"] == 0.0
    validate_contact_metrics(metrics, tolerance=1.0e-4)


def test_contact_matching_preserves_full_tuple_associations():
    mask = torch.tensor([[True, True]])
    dataset_inputs = _inputs(mask)
    runtime_inputs = _inputs(mask)
    dataset_normals = dataset_inputs["contact_normals"].reshape(1, 1, 2, 3)
    runtime_normals = runtime_inputs["contact_normals"].reshape(1, 1, 2, 3)
    dataset_normals[0, 0, 0, 0] = 1.0
    dataset_normals[0, 0, 1, 1] = 1.0
    runtime_normals[0, 0, 0, 1] = 1.0
    runtime_normals[0, 0, 1, 0] = 1.0
    dataset_inputs["contact_depths"][0, 0, 1] = 0.01
    runtime_inputs["contact_depths"][0, 0, 1] = 0.01

    metrics = contact_metrics(dataset_inputs, runtime_inputs, 2)

    with pytest.raises(ValueError, match="matched_depth_abs_max"):
        validate_contact_metrics(metrics, tolerance=1.0e-4)


def test_contact_matching_minimizes_worst_normalized_tuple_error():
    mask = torch.tensor([[True, True]])
    dataset_inputs = _inputs(mask)
    runtime_inputs = _inputs(mask)
    dataset_normals = dataset_inputs["contact_normals"].reshape(1, 1, 2, 3)
    runtime_normals = runtime_inputs["contact_normals"].reshape(1, 1, 2, 3)
    dataset_normals[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
    dataset_normals[0, 0, 1] = torch.tensor([math.cos(1.1e-3), math.sin(1.1e-3), 0.0])
    runtime_normals[0, 0, 0] = dataset_normals[0, 0, 1]
    runtime_normals[0, 0, 1] = torch.tensor([math.cos(0.6e-3), math.sin(0.6e-3), 0.0])
    runtime_inputs["contact_points_0"][0, 0, 3] = 0.6e-4
    runtime_inputs["contact_depths"][0, 0, 1] = 0.6e-4
    runtime_inputs["contact_thicknesses_0"][0, 0, 1] = 0.6e-4
    runtime_inputs["contact_thicknesses_1"][0, 0, 1] = 0.6e-4

    metrics = contact_metrics(dataset_inputs, runtime_inputs, 2)

    assert metrics["matched_normal_l2_max"] == pytest.approx(0.6e-3)
    assert metrics["matched_point0_distance_max"] == pytest.approx(0.6e-4)
    validate_contact_metrics(metrics, tolerance=1.0e-4, normal_tolerance=1.0e-3)


def test_contact_metrics_rejects_true_geometric_difference():
    dataset_inputs = _inputs(torch.tensor([[True, False]]))
    runtime_inputs = _inputs(torch.tensor([[True, True]]))
    runtime_inputs["contact_points_0"][0, 0, 3] = 1.0
    runtime_inputs["contact_points_1"][0, 0, 3] = 1.0

    metrics = contact_metrics(dataset_inputs, runtime_inputs, 2)

    assert metrics["set_hausdorff_distance_max"] == 1.0
    assert math.isinf(metrics["matched_point0_distance_max"])
    with pytest.raises(ValueError, match="set_hausdorff_distance_max"):
        validate_contact_metrics(metrics, tolerance=1.0e-4)


def test_contact_metrics_handles_two_empty_sets():
    inputs = _inputs(torch.tensor([[False, False]]))

    metrics = contact_metrics(inputs, inputs, 2)

    assert metrics["set_hausdorff_distance_max"] == 0.0
    assert metrics["unique_count_abs_diff_max"] == 0.0
    validate_contact_metrics(metrics, tolerance=1.0e-4)


def test_load_random_dataset_batch_respects_history_and_horizon(tmp_path):
    dataset_path = tmp_path / "random_windows.hdf5"
    with h5py.File(dataset_path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        data_group.create_dataset("states", data=np.zeros((3, 20, 4), dtype=np.float32))
        data_group.create_dataset("contact_masks", data=np.zeros((3, 20, 2), dtype=bool))
        data_group.create_dataset("traj_lengths", data=np.asarray([20, 12, 6], dtype=np.int32))
        context_group = dataset_file.create_group("context").create_group("trajectories")
        context_group.create_dataset("terrain_level", data=np.asarray([1, 2, 3], dtype=np.int64))

    batch, metadata = load_random_dataset_batch(
        str(dataset_path),
        num_envs=8,
        history_length=3,
        eval_horizon=5,
        rng=np.random.default_rng(7),
        device="cpu",
    )

    assert batch["states"].shape == (8, 3, 4)
    assert batch["contact_masks"].shape == (8, 3, 2)
    assert set(metadata["trajectory_index"].tolist()).issubset({0, 1})
    for trajectory_index, step in zip(metadata["trajectory_index"], metadata["step"], strict=True):
        assert 2 <= step <= (19 if trajectory_index == 0 else 11) - 5 + 1


def test_load_random_dataset_batch_accepts_exact_fit_window(tmp_path):
    dataset_path = tmp_path / "exact_fit.hdf5"
    with h5py.File(dataset_path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        data_group.create_dataset("states", data=np.zeros((1, 7, 4), dtype=np.float32))
        data_group.create_dataset("traj_lengths", data=np.asarray([7], dtype=np.int32))

    batch, metadata = load_random_dataset_batch(
        str(dataset_path),
        num_envs=1,
        history_length=3,
        eval_horizon=5,
        rng=np.random.default_rng(0),
        device="cpu",
    )

    assert batch["states"].shape == (1, 3, 4)
    assert metadata["step"].tolist() == [2]


def test_load_random_dataset_batch_excludes_previously_sampled_trajectories(tmp_path):
    dataset_path = tmp_path / "unique_windows.hdf5"
    with h5py.File(dataset_path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        data_group.create_dataset("states", data=np.zeros((3, 7, 4), dtype=np.float32))
        data_group.create_dataset("traj_lengths", data=np.asarray([7, 7, 7], dtype=np.int32))

    _, metadata = load_random_dataset_batch(
        str(dataset_path),
        num_envs=2,
        history_length=3,
        eval_horizon=5,
        rng=np.random.default_rng(0),
        device="cpu",
        excluded_trajectory_indices={0},
    )

    assert set(metadata["trajectory_index"].tolist()) == {1, 2}


def test_aggregate_contact_metrics_ignores_undefined_slot_distances():
    base = {
        "mask_mismatch_count": 0.0,
        "mask_total_count": 2.0,
        "mismatched_env_count": 0.0,
        "dataset_active_mean": 1.0,
        "runtime_active_mean": 1.0,
        "active_count_abs_diff_mean": 0.0,
        "slot_point_distance_mean": float("nan"),
        "slot_point_distance_max": float("nan"),
        "slot_point_distance_sum": 0.0,
        "shared_active_slot_count": 0.0,
        "set_chamfer_distance_mean": 0.0,
        "set_hausdorff_distance_mean": 0.0,
        "set_hausdorff_distance_max": 0.0,
        "matched_point0_distance_max": 0.0,
        "matched_normal_l2_max": 0.0,
        "matched_depth_abs_max": 0.0,
        "matched_thickness0_abs_max": 0.0,
        "matched_thickness1_abs_max": 0.0,
        "dataset_unique_active_mean": 1.0,
        "runtime_unique_active_mean": 1.0,
        "unique_count_abs_diff_mean": 0.0,
        "unique_count_abs_diff_max": 0.0,
        "one_sided_empty_count": 0.0,
        "one_sided_empty_fraction": 0.0,
    }
    finite = {
        **base,
        "slot_point_distance_mean": 1.0,
        "slot_point_distance_max": 2.0,
        "slot_point_distance_sum": 1.0,
        "shared_active_slot_count": 1.0,
    }

    forward = aggregate_contact_metrics([base, finite], batch_size=1)
    reverse = aggregate_contact_metrics([finite, base], batch_size=1)

    assert forward["slot_point_distance_mean"] == 1.0
    assert forward["slot_point_distance_max"] == 2.0
    assert reverse["slot_point_distance_mean"] == 1.0
    assert reverse["slot_point_distance_max"] == 2.0


def test_aggregate_contact_metrics_weights_slot_mean_by_shared_contacts():
    base = {
        "mask_mismatch_count": 0.0,
        "mask_total_count": 2.0,
        "mismatched_env_count": 0.0,
        "dataset_active_mean": 1.0,
        "runtime_active_mean": 1.0,
        "active_count_abs_diff_mean": 0.0,
        "slot_point_distance_mean": 1.0,
        "slot_point_distance_max": 1.0,
        "slot_point_distance_sum": 1.0,
        "shared_active_slot_count": 1.0,
        "set_chamfer_distance_mean": 0.0,
        "set_hausdorff_distance_mean": 0.0,
        "set_hausdorff_distance_max": 0.0,
        "matched_point0_distance_max": 0.0,
        "matched_normal_l2_max": 0.0,
        "matched_depth_abs_max": 0.0,
        "matched_thickness0_abs_max": 0.0,
        "matched_thickness1_abs_max": 0.0,
        "dataset_unique_active_mean": 1.0,
        "runtime_unique_active_mean": 1.0,
        "unique_count_abs_diff_mean": 0.0,
        "unique_count_abs_diff_max": 0.0,
        "one_sided_empty_count": 0.0,
        "one_sided_empty_fraction": 0.0,
    }
    three_slots = {
        **base,
        "slot_point_distance_mean": 0.0,
        "slot_point_distance_max": 0.0,
        "slot_point_distance_sum": 0.0,
        "shared_active_slot_count": 3.0,
    }

    summary = aggregate_contact_metrics([base, three_slots], batch_size=1)

    assert summary["slot_point_distance_mean"] == 0.25


def test_aggregate_contact_metrics_preserves_invalid_chamfer():
    metrics = {
        "mask_mismatch_count": 0.0,
        "mask_total_count": 2.0,
        "mismatched_env_count": 0.0,
        "dataset_active_mean": 1.0,
        "runtime_active_mean": 1.0,
        "active_count_abs_diff_mean": 0.0,
        "slot_point_distance_mean": 0.0,
        "slot_point_distance_max": 0.0,
        "slot_point_distance_sum": 0.0,
        "shared_active_slot_count": 1.0,
        "set_chamfer_distance_mean": 0.0,
        "set_hausdorff_distance_mean": 0.0,
        "set_hausdorff_distance_max": 0.0,
        "matched_point0_distance_max": 0.0,
        "matched_normal_l2_max": 0.0,
        "matched_depth_abs_max": 0.0,
        "matched_thickness0_abs_max": 0.0,
        "matched_thickness1_abs_max": 0.0,
        "dataset_unique_active_mean": 1.0,
        "runtime_unique_active_mean": 1.0,
        "unique_count_abs_diff_mean": 0.0,
        "unique_count_abs_diff_max": 0.0,
        "one_sided_empty_count": 0.0,
        "one_sided_empty_fraction": 0.0,
    }
    invalid = {**metrics, "set_chamfer_distance_mean": float("nan")}

    summary = aggregate_contact_metrics([metrics, invalid], batch_size=1)

    assert math.isnan(summary["set_chamfer_distance_mean"])


def test_aggregate_contact_metrics_preserves_active_nan_slot_max():
    metrics = {
        "mask_mismatch_count": 0.0,
        "mask_total_count": 2.0,
        "mismatched_env_count": 0.0,
        "dataset_active_mean": 1.0,
        "runtime_active_mean": 1.0,
        "active_count_abs_diff_mean": 0.0,
        "slot_point_distance_mean": 0.0,
        "slot_point_distance_max": 0.0,
        "slot_point_distance_sum": 0.0,
        "shared_active_slot_count": 1.0,
        "set_chamfer_distance_mean": 0.0,
        "set_hausdorff_distance_mean": 0.0,
        "set_hausdorff_distance_max": 0.0,
        "matched_point0_distance_max": 0.0,
        "matched_normal_l2_max": 0.0,
        "matched_depth_abs_max": 0.0,
        "matched_thickness0_abs_max": 0.0,
        "matched_thickness1_abs_max": 0.0,
        "dataset_unique_active_mean": 1.0,
        "runtime_unique_active_mean": 1.0,
        "unique_count_abs_diff_mean": 0.0,
        "unique_count_abs_diff_max": 0.0,
        "one_sided_empty_count": 0.0,
        "one_sided_empty_fraction": 0.0,
    }
    invalid = {**metrics, "slot_point_distance_max": float("nan")}

    summary = aggregate_contact_metrics([invalid, metrics], batch_size=1)

    assert math.isnan(summary["slot_point_distance_max"])
