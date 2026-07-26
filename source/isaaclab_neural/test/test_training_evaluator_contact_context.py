# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for strict rollout-evaluator contact diagnostics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_neural.eval.training_evaluator import TrainingRolloutEvaluator


def _contacts(mask: torch.Tensor, points: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    batch_size, num_contacts = mask.shape
    if points is None:
        points = torch.zeros(batch_size, num_contacts, 3)
    return {
        "contact_masks": mask,
        "contact_points_0": points.reshape(batch_size, -1).clone(),
        "contact_points_1": points.reshape(batch_size, -1).clone(),
        "contact_normals": torch.zeros(batch_size, num_contacts * 3),
        "contact_depths": torch.zeros(batch_size, num_contacts),
        "contact_thicknesses_0": torch.zeros(batch_size, num_contacts),
        "contact_thicknesses_1": torch.zeros(batch_size, num_contacts),
    }


def _evaluator(
    runtime_contacts: dict[str, torch.Tensor],
    *,
    contact_context_validation: str = "warn",
) -> TrainingRolloutEvaluator:
    evaluator = object.__new__(TrainingRolloutEvaluator)
    evaluator.require_terrain_context = True
    evaluator.contact_context_validation = contact_context_validation
    evaluator.state_context_tolerance = 1.0e-5
    evaluator.contact_context_tolerance = 1.0e-4
    evaluator.contact_context_normal_tolerance = 1.0e-3
    evaluator.neural_env = SimpleNamespace(
        solver_neural=SimpleNamespace(
            contacts=runtime_contacts,
            states=torch.zeros(2, 4),
            root_body_q=torch.zeros(2, 7),
        )
    )
    return evaluator


def _trajectories(recorded_contacts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    trajectories = {
        key: value.unsqueeze(1) for key, value in recorded_contacts.items()
    }
    trajectories.update(
        {
        "_dataset_trajectory_index": torch.tensor([11, 22]),
        "_dataset_window_start": torch.tensor([30, 40]),
        "terrain_level": torch.tensor([1, 2]),
        "terrain_type": torch.tensor([3, 4]),
        }
    )
    return trajectories


@pytest.mark.parametrize(
    ("contact_mode", "expected_validation"),
    (("newton_native", "warn"), ("fixed_ground", "strict")),
)
def test_contact_validation_default_depends_on_contact_mode(contact_mode, expected_validation):
    neural_env = SimpleNamespace(
        solver_neural=SimpleNamespace(
            contact_mode=contact_mode,
            num_states_history=1,
        )
    )

    evaluator = TrainingRolloutEvaluator(neural_env)

    assert evaluator.contact_context_validation == expected_validation


def test_runtime_contact_mismatch_warns_with_dataset_and_terrain_coordinates(caplog):
    runtime_mask = torch.tensor([[True, False, False], [True, False, False]])
    recorded_mask = torch.tensor([[True, False, False], [True, True, False]])
    recorded_points = torch.zeros(2, 3, 3)
    recorded_points[1, 1, 0] = 1.0

    _evaluator(_contacts(runtime_mask))._validate_runtime_contact_context(
        _trajectories(_contacts(recorded_mask, recorded_points)),
        start=0,
        end=2,
        history_offset=0,
    )

    message = caplog.text
    assert "mismatches=1/6" in message
    assert "trajectory=22" in message
    assert "step=40" in message
    assert "terrain=(2, 4)" in message
    assert "slots=[1]" in message


def test_runtime_contact_match_passes_strict_validation():
    mask = torch.tensor([[True, False, False], [True, True, False]])

    _evaluator(_contacts(mask))._validate_runtime_contact_context(
        _trajectories(_contacts(mask)),
        start=0,
        end=2,
        history_offset=0,
    )


def test_runtime_contact_validation_rejects_nonfinite_float_fields():
    mask = torch.tensor([[True, False, False], [True, True, False]])
    evaluator = _evaluator(_contacts(mask))
    evaluator.neural_env.solver_neural.contacts["contact_depths"][1, 0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        evaluator._validate_runtime_contact_context(
            _trajectories(_contacts(mask)),
            start=0,
            end=2,
            history_offset=0,
        )


def test_runtime_contact_validation_rejects_missing_dataset_fields():
    mask = torch.tensor([[True, False, False], [True, True, False]])
    trajectories = _trajectories(_contacts(mask))
    del trajectories["contact_depths"]

    with pytest.raises(ValueError, match="Dataset contact context is missing"):
        _evaluator(_contacts(mask))._validate_runtime_contact_context(
            trajectories,
            start=0,
            end=2,
            history_offset=0,
        )


def test_runtime_contact_validation_warns_for_slot_permutation(caplog):
    mask = torch.tensor([[True, True, False], [False, False, False]])
    dataset_points = torch.zeros(2, 3, 3)
    runtime_points = torch.zeros(2, 3, 3)
    dataset_points[0, 0, 0] = 1.0
    runtime_points[0, 1, 0] = 1.0

    _evaluator(_contacts(mask, runtime_points))._validate_runtime_contact_context(
        _trajectories(_contacts(mask, dataset_points)),
        start=0,
        end=2,
        history_offset=0,
    )

    assert "unordered contact sets matched" in caplog.text
    assert "slot_point1_max=1" in caplog.text
    assert "hausdorff_max=0" in caplog.text


def test_runtime_contact_validation_warns_for_duplicate_contact(caplog):
    dataset_mask = torch.tensor([[True, False, False], [False, False, False]])
    runtime_mask = torch.tensor([[True, True, False], [False, False, False]])
    runtime_points = torch.zeros(2, 3, 3)
    runtime_points[0, 1, 0] = 5.0e-5

    _evaluator(_contacts(runtime_mask, runtime_points))._validate_runtime_contact_context(
        _trajectories(_contacts(dataset_mask)),
        start=0,
        end=2,
        history_offset=0,
    )

    assert "mask_mismatches=1/6" in caplog.text
    assert "unique_count_diff_max=0" in caplog.text


def test_runtime_contact_validation_warns_for_true_set_difference(caplog):
    dataset_mask = torch.tensor([[True, False, False], [False, False, False]])
    runtime_mask = torch.tensor([[True, True, False], [False, False, False]])
    runtime_points = torch.zeros(2, 3, 3)
    runtime_points[0, 1, 0] = 1.0

    _evaluator(_contacts(runtime_mask, runtime_points))._validate_runtime_contact_context(
        _trajectories(_contacts(dataset_mask)),
        start=0,
        end=2,
        history_offset=0,
    )

    assert "set_hausdorff_distance_max" in caplog.text
    assert "diagnostic-only" in caplog.text


def test_runtime_contact_validation_strict_mode_rejects_true_set_difference():
    dataset_mask = torch.tensor([[True, False, False], [False, False, False]])
    runtime_mask = torch.tensor([[True, True, False], [False, False, False]])
    runtime_points = torch.zeros(2, 3, 3)
    runtime_points[0, 1, 0] = 1.0

    with pytest.raises(ValueError, match="set_hausdorff_distance_max"):
        _evaluator(
            _contacts(runtime_mask, runtime_points),
            contact_context_validation="strict",
        )._validate_runtime_contact_context(
            _trajectories(_contacts(dataset_mask)),
            start=0,
            end=2,
            history_offset=0,
        )


def test_runtime_contact_validation_handles_empty_sets():
    mask = torch.zeros(2, 3, dtype=torch.bool)

    _evaluator(_contacts(mask))._validate_runtime_contact_context(
        _trajectories(_contacts(mask)),
        start=0,
        end=2,
        history_offset=0,
    )


def test_runtime_state_context_rejects_reset_mismatch():
    mask = torch.zeros(2, 3, dtype=torch.bool)
    evaluator = _evaluator(_contacts(mask))
    trajectories = _trajectories(_contacts(mask))
    trajectories["states"] = torch.zeros(2, 1, 4)
    trajectories["root_body_q"] = torch.zeros(2, 1, 7)
    evaluator.neural_env.solver_neural.states[1, 0] = 1.0e-3

    with pytest.raises(ValueError, match="states_max_error"):
        evaluator._validate_runtime_state_context(
            trajectories,
            start=0,
            end=2,
            history_offset=0,
        )


def test_runtime_state_context_rejects_root_pose_mismatch():
    mask = torch.zeros(2, 3, dtype=torch.bool)
    evaluator = _evaluator(_contacts(mask))
    trajectories = _trajectories(_contacts(mask))
    trajectories["states"] = torch.zeros(2, 1, 4)
    trajectories["root_body_q"] = torch.zeros(2, 1, 7)
    evaluator.neural_env.solver_neural.root_body_q[0, 2] = 1.0e-3

    with pytest.raises(ValueError, match="root_body_q_max_error"):
        evaluator._validate_runtime_state_context(
            trajectories,
            start=0,
            end=2,
            history_offset=0,
        )


def test_strict_history_context_requires_preload_support():
    evaluator = _evaluator(_contacts(torch.zeros(2, 3, dtype=torch.bool)))

    with pytest.raises(ValueError, match="history preloading support"):
        evaluator._preload_history(
            trajectories={},
            start=0,
            end=2,
            history_offset=1,
        )


def test_sampled_trajectories_include_source_window_coordinates(monkeypatch):
    class FakeDataset:
        mapping_index2traj = np.asarray([[5, 10], [6, 20], [7, 30]])

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"states": torch.tensor([float(index)])}

    evaluator = object.__new__(TrainingRolloutEvaluator)
    evaluator.trajectory_dataset = FakeDataset()
    monkeypatch.setattr(np.random, "randint", lambda **_kwargs: np.asarray([2, 0]))

    trajectories = evaluator._sample_dataset_trajectories(2)

    torch.testing.assert_close(trajectories["_dataset_trajectory_index"], torch.tensor([7, 5]))
    torch.testing.assert_close(trajectories["_dataset_window_start"], torch.tensor([30, 10]))
