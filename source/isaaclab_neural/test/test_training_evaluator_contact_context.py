# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for strict rollout-evaluator contact diagnostics."""

from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest
import torch
from isaaclab_neural.data import TrajectoryDataset
from isaaclab_neural.data.hdf5 import write_rollouts_to_hdf5
from isaaclab_neural.eval.training_evaluator import TrainingRolloutEvaluator
from isaaclab_neural.train import trainers as trainers_module
from native_hdf5_samples import make_native_rollouts


class _EvaluatorTestHarness(TrainingRolloutEvaluator):
    rebase_replica_trajectories = TrainingRolloutEvaluator._rebase_replica_trajectories
    validate_contact_context = TrainingRolloutEvaluator._validate_runtime_contact_context
    validate_state_context = TrainingRolloutEvaluator._validate_runtime_state_context
    preload_history = TrainingRolloutEvaluator._preload_history
    history_offset = TrainingRolloutEvaluator._history_offset
    sample_dataset_trajectories = TrainingRolloutEvaluator._sample_dataset_trajectories
    restore_terrain_context = TrainingRolloutEvaluator._restore_terrain_context
    source_origin_batches = TrainingRolloutEvaluator._source_origin_batches
    validate_runtime_contact_tokens = TrainingRolloutEvaluator._validate_runtime_contact_tokens
    evaluate_control_rollout = TrainingRolloutEvaluator._evaluate_control_rollout


class _TrainerTestHarness(trainers_module.VanillaTrainer):
    initialize_evaluator = trainers_module.VanillaTrainer._init_evaluator


def _contacts(mask: torch.Tensor, points: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    batch_size, num_contacts = mask.shape
    contact_points: torch.Tensor = torch.zeros(batch_size, num_contacts, 3) if points is None else points
    return {
        "contact_masks": mask,
        "contact_points_0": contact_points.reshape(batch_size, -1).clone(),
        "contact_points_1": contact_points.reshape(batch_size, -1).clone(),
        "contact_normals": torch.zeros(batch_size, num_contacts * 3),
        "contact_depths": torch.zeros(batch_size, num_contacts),
        "contact_thicknesses_0": torch.zeros(batch_size, num_contacts),
        "contact_thicknesses_1": torch.zeros(batch_size, num_contacts),
    }


def _token_contacts(tokens: torch.Tensor, overflow: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if overflow is None:
        overflow = torch.zeros(tokens.shape[0], dtype=torch.long)
    return {
        "contact_tokens": tokens,
        "contact_token_overflow": overflow,
    }


def _evaluator(
    runtime_contacts: dict[str, torch.Tensor],
    *,
    contact_context_validation: str = "warn",
) -> _EvaluatorTestHarness:
    evaluator = object.__new__(_EvaluatorTestHarness)
    evaluator.require_terrain_context = True
    evaluator.contact_context_validation = contact_context_validation
    evaluator.state_context_tolerance = 1.0e-5
    evaluator.contact_context_tolerance = 1.0e-4
    evaluator.contact_context_normal_tolerance = 1.0e-3
    evaluator.contact_context_velocity_tolerance = 1.0e-3
    evaluator.neural_env = SimpleNamespace(
        solver_neural=SimpleNamespace(
            contact_inputs=runtime_contacts,
            contact_representation="contact_tokens" if "contact_tokens" in runtime_contacts else "flat",
            states=torch.zeros(2, 4),
            root_body_q=torch.zeros(2, 7),
            root_body_qd=torch.zeros(2, 6),
        )
    )
    return evaluator


def _trajectories(recorded_contacts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    trajectories = {key: value.unsqueeze(1) for key, value in recorded_contacts.items()}
    trajectories.update(
        {
            "_dataset_trajectory_index": torch.tensor([11, 22]),
            "_dataset_window_start": torch.tensor([30, 40]),
            "terrain_level": torch.tensor([1, 2]),
            "terrain_type": torch.tensor([3, 4]),
        }
    )
    return trajectories


def test_replica_rebase_moves_free_objects_to_the_runtime_table() -> None:
    from isaaclab_neural.solvers.neural_solver import NeuralSolver

    solver = NeuralSolver.__new__(NeuralSolver)
    solver.contact_mode = "newton_native"
    solver.free_joint_indices = np.array([1])
    solver.joint_q_start = np.array([0, 1])
    scene = SimpleNamespace(env_origins=torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]))
    env = SimpleNamespace(
        solver_neural=solver,
        num_envs=2,
        neural_adapter=SimpleNamespace(isaaclab_env=SimpleNamespace(scene=scene)),
    )
    evaluator = _EvaluatorTestHarness(env, device="cpu", rebase_env_origins=True)
    source_origins = torch.tensor([[20.0, 0.0, 0.0], [-10.0, 5.0, 0.0], [5.0, -5.0, 0.0], [0.0, 10.0, 0.0]])
    states = torch.zeros(4, 2, 15)
    states[..., 0] = 0.2
    states[..., 1:4] = source_origins[:, None, :] + torch.tensor([0.5, 0.0, 0.025])
    trajectories = {"states": states, "next_states": states.clone(), "env_origin": source_origins}
    rebased = evaluator.rebase_replica_trajectories(trajectories)
    expected_origins = scene.env_origins.repeat(2, 1)
    torch.testing.assert_close(rebased["env_origin"], expected_origins)
    torch.testing.assert_close(
        rebased["states"][..., 1:4],
        (expected_origins[:, None, :] + torch.tensor([0.5, 0.0, 0.025])).expand(4, 2, 3),
    )
    torch.testing.assert_close(rebased["states"][..., 0], states[..., 0])
    torch.testing.assert_close(rebased["next_states"], rebased["states"])
    torch.testing.assert_close(scene.env_origins, torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]))
    assert trajectories["states"] is states
    with pytest.raises(ValueError, match="env_origin"):
        evaluator.rebase_replica_trajectories({"states": states})
    with pytest.raises(ValueError, match="terrain"):
        TrainingRolloutEvaluator(env, rebase_env_origins=True, require_terrain_context=True)


def test_source_origin_batches_preserve_coordinates_and_split_duplicates():
    origins = torch.tensor([[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    env = SimpleNamespace(
        num_envs=3,
        neural_adapter=SimpleNamespace(isaaclab_env=SimpleNamespace(scene=SimpleNamespace(env_origins=origins))),
    )
    evaluator = _EvaluatorTestHarness.__new__(_EvaluatorTestHarness)
    evaluator.neural_env = env
    ids = torch.tensor([2, 0, 2, 1])
    data = {"states": torch.zeros(4, 2, 1), "source_env_id": ids, "env_origin": origins[ids].clone()}
    batches = evaluator.source_origin_batches(data)
    assert [(indices.tolist(), slots.tolist()) for indices, slots in batches] == [([0, 1, 3], [2, 0, 1]), ([2], [2])]
    torch.testing.assert_close(data["env_origin"], origins[ids])
    with pytest.raises(ValueError, match="source_env_id"):
        evaluator.source_origin_batches({k: v for k, v in data.items() if k != "source_env_id"})
    with pytest.raises(ValueError, match="original.*grid"):
        evaluator.source_origin_batches({**data, "env_origin": data["env_origin"] + 0.001})
    with pytest.raises(ValueError, match="source_env_id"):
        evaluator.source_origin_batches({**data, "source_env_id": torch.tensor([3, 0, 2, 1])})


def test_source_origin_rollout_retains_duplicate_and_partial_batches(monkeypatch):
    origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    contacts = _token_contacts(torch.zeros(3, 1, 17))
    solver = SimpleNamespace(
        num_states_history=1,
        contact_mode="newton_native",
        contact_representation="contact_tokens",
        state_dim=1,
        states=torch.zeros(3, 1),
        root_body_q=torch.zeros(3, 7),
        root_body_qd=torch.zeros(3, 6),
        joint_f=torch.zeros(3, 1),
        gravity_dir=torch.zeros(3, 3),
        contact_inputs=contacts,
        eval=lambda: None,
    )

    def reset(value=None):
        solver.states = torch.zeros(3, 1) if value is None else value.clone()

    adapter = SimpleNamespace(reset=reset, isaaclab_env=SimpleNamespace(scene=SimpleNamespace(env_origins=origins)))
    env = SimpleNamespace(solver_neural=solver, num_envs=3, neural_adapter=adapter)
    evaluator = _EvaluatorTestHarness(env, eval_horizon=2, device="cpu", preserve_source_origins=True)
    calls = []
    monkeypatch.setattr(evaluator, "_validate_runtime_state_context", lambda *a, **k: None)
    monkeypatch.setattr(
        evaluator, "_validate_runtime_contact_tokens", lambda *a, **k: calls.append(k["env_ids"].tolist())
    )
    monkeypatch.setattr(evaluator, "calculate_error_metrics", lambda target, actual: (actual - target, {}))
    ids = torch.tensor([2, 0, 2, 1])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    data = {
        "states": values[:, None, None].expand(4, 2, 1).clone(),
        "next_states": values[:, None, None] + torch.tensor([1.0, 2.0])[None, :, None],
        "joint_f": torch.ones(4, 2, 1),
        "source_env_id": ids,
        "env_origin": origins[ids],
    }

    def step(force):
        solver.states += force
        return solver.states.clone()

    difference, output, _ = evaluator.evaluate_control_rollout(
        data, control_key="joint_f", step_fn=step, eval_mode="rollout", render=False
    )
    assert calls == [[2, 0, 1], [2]]
    torch.testing.assert_close(difference, torch.zeros(4, 2, 1))
    torch.testing.assert_close(output["states"][:, 0, 0], values)
    assert output["states"].shape == (4, 3, 1)


def test_source_validation_checks_only_active_slots_and_rejects_bad_contacts():
    tokens = torch.zeros(2, 1, 17)
    tokens[..., 0] = 1
    tokens[..., 1:4] = torch.tensor([9.0, 11.0, 1.0])
    tokens[..., 9] = 1
    runtime = tokens.clone()
    runtime[1, 0, 4] = 0.01
    evaluator = _evaluator(_token_contacts(runtime), contact_context_validation="strict")
    trajectories = _trajectories(_token_contacts(tokens))
    evaluator.validate_runtime_contact_tokens(trajectories, 0, 2, 0, env_ids=torch.tensor([0]))
    with pytest.raises(ValueError, match="trajectory=22"):
        evaluator.validate_runtime_contact_tokens(trajectories, 0, 2, 0, env_ids=torch.tensor([1]))


def test_contact_validation_defaults_to_warn():
    neural_env = SimpleNamespace(
        solver_neural=SimpleNamespace(
            contact_mode="newton_native",
            num_states_history=1,
        )
    )

    evaluator = _EvaluatorTestHarness(neural_env)

    assert evaluator.contact_context_validation == "warn"


def test_contact_validation_rejects_legacy_mode():
    neural_env = SimpleNamespace(
        solver_neural=SimpleNamespace(
            contact_mode="fixed_ground",
            num_states_history=1,
        )
    )

    with pytest.warns(DeprecationWarning, match="fixed_ground"):
        with pytest.raises(ValueError, match="incompatible contact representations"):
            TrainingRolloutEvaluator(neural_env)


def test_runtime_contact_mismatch_warns_with_dataset_and_terrain_coordinates(caplog):
    runtime_mask = torch.tensor([[True, False, False], [True, False, False]])
    recorded_mask = torch.tensor([[True, False, False], [True, True, False]])
    recorded_points = torch.zeros(2, 3, 3)
    recorded_points[1, 1, 0] = 1.0

    _evaluator(_contacts(runtime_mask)).validate_contact_context(
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

    _evaluator(_contacts(mask)).validate_contact_context(
        _trajectories(_contacts(mask)),
        start=0,
        end=2,
        history_offset=0,
    )


def test_runtime_contact_validation_rejects_nonfinite_float_fields():
    mask = torch.tensor([[True, False, False], [True, True, False]])
    evaluator = _evaluator(_contacts(mask))
    evaluator.neural_env.solver_neural.contact_inputs["contact_depths"][1, 0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        evaluator.validate_contact_context(
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
        _evaluator(_contacts(mask)).validate_contact_context(
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

    _evaluator(_contacts(mask, runtime_points)).validate_contact_context(
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

    _evaluator(_contacts(runtime_mask, runtime_points)).validate_contact_context(
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

    _evaluator(_contacts(runtime_mask, runtime_points)).validate_contact_context(
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
        ).validate_contact_context(
            _trajectories(_contacts(dataset_mask)),
            start=0,
            end=2,
            history_offset=0,
        )


def test_runtime_contact_validation_handles_empty_sets():
    mask = torch.zeros(2, 3, dtype=torch.bool)

    _evaluator(_contacts(mask)).validate_contact_context(
        _trajectories(_contacts(mask)),
        start=0,
        end=2,
        history_offset=0,
    )


def test_runtime_contact_token_mismatch_is_diagnostic_only(caplog):
    dataset_tokens = torch.zeros(2, 3, 17)
    runtime_tokens = dataset_tokens.clone()
    dataset_tokens[0, 0, 0] = 1.0
    runtime_tokens[0, 0, 0] = 1.0
    runtime_tokens[0, 0, 4] = 0.1

    _evaluator(_token_contacts(runtime_tokens)).validate_contact_context(
        _trajectories(_token_contacts(dataset_tokens)),
        start=0,
        end=2,
        history_offset=0,
    )

    assert "Runtime contact-token context mismatch" in caplog.text
    assert "point_l2_max=0.1" in caplog.text
    assert "diagnostic-only" in caplog.text


def test_runtime_contact_token_strict_mode_accepts_slot_permutation():
    dataset_tokens = torch.zeros(2, 3, 17)
    runtime_tokens = dataset_tokens.clone()
    dataset_tokens[0, 0, 0] = 1.0
    runtime_tokens[0, 1, 0] = 1.0

    _evaluator(
        _token_contacts(runtime_tokens),
        contact_context_validation="strict",
    ).validate_contact_context(
        _trajectories(_token_contacts(dataset_tokens)),
        start=0,
        end=2,
        history_offset=0,
    )


def test_runtime_contact_token_validation_rejects_nonfinite_values():
    tokens = torch.zeros(2, 3, 17)
    runtime_tokens = tokens.clone()
    runtime_tokens[0, 0, 4] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        _evaluator(_token_contacts(runtime_tokens)).validate_contact_context(
            _trajectories(_token_contacts(tokens)),
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
    trajectories["root_body_qd"] = torch.zeros(2, 1, 6)
    evaluator.neural_env.solver_neural.states[1, 0] = 1.0e-3

    with pytest.raises(ValueError, match="states_max_error"):
        evaluator.validate_state_context(
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
    trajectories["root_body_qd"] = torch.zeros(2, 1, 6)
    evaluator.neural_env.solver_neural.root_body_q[0, 2] = 1.0e-3

    with pytest.raises(ValueError, match="root_body_q_max_error"):
        evaluator.validate_state_context(
            trajectories,
            start=0,
            end=2,
            history_offset=0,
        )


def test_runtime_state_context_accepts_equivalent_root_quaternion_sign():
    mask = torch.zeros(2, 3, dtype=torch.bool)
    evaluator = _evaluator(_contacts(mask))
    trajectories = _trajectories(_contacts(mask))
    trajectories["states"] = torch.zeros(2, 1, 4)
    trajectories["root_body_q"] = torch.zeros(2, 1, 7)
    trajectories["root_body_qd"] = torch.zeros(2, 1, 6)
    trajectories["root_body_q"][:, :, 6] = -1.0
    evaluator.neural_env.solver_neural.root_body_q[:, 6] = 1.0

    evaluator.validate_state_context(
        trajectories,
        start=0,
        end=2,
        history_offset=0,
    )


def test_runtime_state_context_rejects_root_velocity_mismatch():
    mask = torch.zeros(2, 3, dtype=torch.bool)
    evaluator = _evaluator(_contacts(mask))
    trajectories = _trajectories(_contacts(mask))
    trajectories["states"] = torch.zeros(2, 1, 4)
    trajectories["root_body_q"] = torch.zeros(2, 1, 7)
    trajectories["root_body_qd"] = torch.zeros(2, 1, 6)
    evaluator.neural_env.solver_neural.root_body_qd[0, 2] = 1.0e-3

    with pytest.raises(ValueError, match="root_body_qd_max_error"):
        evaluator.validate_state_context(
            trajectories,
            start=0,
            end=2,
            history_offset=0,
        )


def test_strict_history_context_requires_preload_support():
    evaluator = _evaluator(_contacts(torch.zeros(2, 3, dtype=torch.bool)))

    with pytest.raises(ValueError, match="history preloading support"):
        evaluator.preload_history(
            trajectories={},
            start=0,
            end=2,
            history_offset=1,
        )


def test_owner_frame_history_preload_requires_primary_body_q():
    tokens = torch.zeros(2, 4, 17)
    evaluator = _evaluator(_token_contacts(tokens))
    evaluator.history_length = 3
    evaluator.device = "cpu"
    evaluator.neural_env.solver_neural.contact_token_frame = "owner"
    evaluator.neural_env.solver_neural.num_envs = 2
    evaluator.neural_env.solver_neural.num_states_history = 3
    evaluator.neural_env.solver_neural.root_body_q = torch.zeros(2, 7)
    evaluator.neural_env.solver_neural.states = torch.zeros(2, 4)
    evaluator.neural_env.solver_neural.joint_f = torch.zeros(2, 3)
    evaluator.neural_env.solver_neural.gravity_dir = torch.zeros(2, 3)
    evaluator.neural_env.solver_neural.primary_body_q = torch.zeros(2, 17, 7)

    trajectories = _trajectories(_token_contacts(tokens))
    trajectories["root_body_q"] = torch.zeros(2, 3, 7)
    trajectories["states"] = torch.zeros(2, 3, 4)
    trajectories["joint_f"] = torch.zeros(2, 3, 3)
    trajectories["gravity_dir"] = torch.zeros(2, 3, 3)
    trajectories["contact_tokens"] = torch.zeros(2, 3, 4, 17)
    trajectories["contact_token_overflow"] = torch.zeros(2, 3, dtype=torch.long)

    with pytest.raises(ValueError, match="primary_body_q"):
        evaluator.history_offset(trajectories)

    captured = {}

    def _preload(history):
        captured.update(history)

    evaluator.neural_env.solver_neural.preload_states_history = _preload
    trajectories["primary_body_q"] = torch.zeros(2, 3, 17, 7)
    assert evaluator.history_offset(trajectories) == 2
    evaluator.preload_history(trajectories, start=0, end=2, history_offset=2)
    assert "primary_body_q" in captured
    assert captured["primary_body_q"].shape == (2, 2, 17, 7)


def test_strict_eval_projects_flat_contact_history_from_native_dataset(tmp_path):
    dataset_path = tmp_path / "native.hdf5"
    write_rollouts_to_hdf5(
        dataset_path,
        make_native_rollouts(num_trajectories=2, num_steps=6),
        env_name="Test-Env",
    )
    neural_env = SimpleNamespace(
        solver_neural=SimpleNamespace(
            contact_mode="newton_native",
            num_states_history=3,
            contact_inputs=_contacts(torch.zeros(1, 2, dtype=torch.bool)),
        )
    )

    evaluator = _EvaluatorTestHarness(
        neural_env,
        hdf5_dataset_path=str(dataset_path),
        eval_horizon=2,
        require_terrain_context=True,
        contact_representation="flat",
        num_contacts_per_env=2,
        contact_packing_policy="penetration_priority",
        contact_filter="solver_active",
    )
    trajectories = evaluator.sample_dataset_trajectories(1)

    assert set(neural_env.solver_neural.contact_inputs).issubset(trajectories)
    assert evaluator.history_offset(trajectories) == 2


def test_trainer_forwards_native_contact_projection_to_evaluator(monkeypatch):
    captured_kwargs = {}

    class FakeEvaluator:
        def __init__(self, *_args, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(trainers_module, "TrainingRolloutEvaluator", FakeEvaluator)
    trainer = object.__new__(_TrainerTestHarness)
    trainer.neural_env = SimpleNamespace(num_envs=1)
    trainer.device = "cpu"
    trainer.dataset_contact_projection = {
        "contact_representation": "flat",
        "num_contacts_per_env": 64,
        "max_contact_tokens": 0,
        "contact_packing_policy": "penetration_priority",
        "contact_filter": "solver_active",
    }

    trainer.initialize_evaluator(
        {
            "eval": {
                "interval": 1,
                "mode": "dataset",
                "action_mode": "joint_f",
                "dataset_path": "native.hdf5",
            }
        },
        {},
    )

    assert {
        key: captured_kwargs[key] for key in trainer.dataset_contact_projection
    } == trainer.dataset_contact_projection


def test_sampled_trajectories_include_source_window_coordinates(monkeypatch):
    dataset = create_autospec(TrajectoryDataset, instance=True)
    dataset.mapping_index2traj = np.asarray([[5, 10], [6, 20], [7, 30]])
    dataset.__len__.return_value = 3
    dataset.__getitem__.side_effect = lambda index: {"states": torch.tensor([float(index)])}
    evaluator = object.__new__(_EvaluatorTestHarness)
    evaluator.trajectory_dataset = dataset
    monkeypatch.setattr(np.random, "randint", lambda **_kwargs: np.asarray([2, 0]))

    trajectories = evaluator.sample_dataset_trajectories(2)

    torch.testing.assert_close(trajectories["_dataset_trajectory_index"], torch.tensor([7, 5]))
    torch.testing.assert_close(trajectories["_dataset_window_start"], torch.tensor([30, 10]))


def _terrain_restore_evaluator(*, require_terrain_context: bool, terrain) -> _EvaluatorTestHarness:
    evaluator = object.__new__(_EvaluatorTestHarness)
    evaluator.require_terrain_context = require_terrain_context
    evaluator.neural_env = SimpleNamespace(
        neural_adapter=SimpleNamespace(isaaclab_env=SimpleNamespace(scene=SimpleNamespace(terrain=terrain)))
    )
    return evaluator


def test_restore_terrain_context_skips_levels_on_flat_terrain():
    terrain = SimpleNamespace(
        device=torch.device("cpu"),
        env_origins=torch.zeros(2, 3),
    )
    trajectories = {
        "terrain_level": torch.tensor([-1, -1]),
        "terrain_type": torch.tensor([-1, -1]),
        "env_origin": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    }

    _terrain_restore_evaluator(require_terrain_context=False, terrain=terrain).restore_terrain_context(
        trajectories,
        start=0,
        end=2,
    )

    torch.testing.assert_close(terrain.env_origins, trajectories["env_origin"])


def test_restore_terrain_context_writes_curriculum_attributes():
    terrain = SimpleNamespace(
        device=torch.device("cpu"),
        terrain_levels=torch.zeros(2, dtype=torch.long),
        terrain_types=torch.zeros(2, dtype=torch.long),
        env_origins=torch.zeros(2, 3),
    )
    trajectories = {
        "terrain_level": torch.tensor([1, 2]),
        "terrain_type": torch.tensor([3, 4]),
        "env_origin": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    }

    _terrain_restore_evaluator(require_terrain_context=True, terrain=terrain).restore_terrain_context(
        trajectories,
        start=0,
        end=2,
    )

    torch.testing.assert_close(terrain.terrain_levels, trajectories["terrain_level"])
    torch.testing.assert_close(terrain.terrain_types, trajectories["terrain_type"])
    torch.testing.assert_close(terrain.env_origins, trajectories["env_origin"])


def test_restore_terrain_context_strict_rejects_flat_terrain():
    terrain = SimpleNamespace(
        device=torch.device("cpu"),
        env_origins=torch.zeros(2, 3),
    )
    trajectories = {
        "terrain_level": torch.tensor([-1, -1]),
        "terrain_type": torch.tensor([-1, -1]),
        "env_origin": torch.zeros(2, 3),
    }

    with pytest.raises(ValueError, match="curriculum terrain attributes"):
        _terrain_restore_evaluator(require_terrain_context=True, terrain=terrain).restore_terrain_context(
            trajectories,
            start=0,
            end=2,
        )
