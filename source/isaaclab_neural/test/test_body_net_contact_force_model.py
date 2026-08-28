# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the optional per-body resultant-contact-force model output."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from isaaclab_neural.models.models import ModelMixedInput, ModelMixedInputOutput
from isaaclab_neural.solvers.neural_solver import NeuralSolver
from isaaclab_neural.train.trainers import (
    BODY_FORCE_LABEL_KEY,
    BODY_FORCE_TARGET_KEY,
    VanillaTrainer,
    body_net_contact_force_metrics,
    transform_body_net_contact_forces_to_current_root_frame,
)
from isaaclab_neural.utils.checkpoint import reconstruct_model_from_checkpoint
from isaaclab_neural.utils.running_mean_std import RunningMeanStd


def _network_cfg(*, force_head: bool | None = None) -> dict:
    cfg = {
        "normalize_input": False,
        "normalize_output": False,
        "encoder": {
            "low_dim": {
                "layer_sizes": [],
                "activation": "relu",
                "layernorm": False,
            }
        },
        "transformer": {
            "block_size": 8,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 8,
            "dropout": 0.0,
            "bias": False,
        },
        "model": {
            "mlp": {
                "layer_sizes": [6],
                "activation": "relu",
                "layernorm": False,
            }
        },
    }
    if force_head is not None:
        cfg["body_net_contact_force_head"] = {
            "enabled": force_head,
            "model": {
                "mlp": {
                    "layer_sizes": [5],
                    "activation": "relu",
                    "layernorm": False,
                }
            },
        }
    return cfg


def _model(*, force_head: bool | None = None, num_bodies: int = 7) -> ModelMixedInput:
    return ModelMixedInput(
        input_sample={"states_embedding": torch.zeros(2, 4, 4)},
        output_dim=9,
        input_cfg={"low_dim": ["states_embedding"]},
        network_cfg=_network_cfg(force_head=force_head),
        num_bodies=num_bodies,
        device="cpu",
    )


def test_auxiliary_force_shape_and_single_shared_trunk_pass():
    model = _model(force_head=True, num_bodies=7)
    trunk_calls = 0

    def count_trunk_calls(_module, _inputs):
        nonlocal trunk_calls
        trunk_calls += 1

    hook = model.transformer_model.register_forward_pre_hook(count_trunk_calls)
    output = model.forward_with_aux({"states_embedding": torch.randn(2, 4, 4)})
    hook.remove()

    assert output.state_prediction.shape == (2, 4, 9)
    assert output.auxiliary_predictions["body_net_contact_forces_b"].shape == (2, 4, 7, 3)
    assert trunk_calls == 1
    assert model({"states_embedding": torch.randn(2, 4, 4)}).shape == (2, 4, 9)


def test_auxiliary_force_gradients_reach_head_and_shared_trunk():
    model = _model(force_head=True)
    output = model.forward_with_aux({"states_embedding": torch.randn(2, 4, 4)})
    loss = output.state_prediction.square().mean()
    loss += output.auxiliary_predictions["body_net_contact_forces_b"].square().mean()
    loss.backward()

    assert model.body_net_contact_force_head.output_net.weight.grad is not None
    assert model.transformer_model.transformer.wte.weight.grad is not None
    assert torch.count_nonzero(model.transformer_model.transformer.wte.weight.grad) > 0


def test_disabled_head_strictly_loads_old_state_only_state_dict():
    old_model = _model()
    disabled_model = _model(force_head=False)

    assert old_model.state_dict().keys() == disabled_model.state_dict().keys()
    disabled_model.load_state_dict(old_model.state_dict(), strict=True)
    assert not hasattr(disabled_model, "body_net_contact_force_head")
    assert disabled_model.forward_with_aux({"states_embedding": torch.randn(2, 4, 4)}).auxiliary_predictions == {}


class _CheckpointSolver:
    prediction_dim = 9
    contact_mode = "newton_native"
    num_contact_bodies_per_env = 5

    @staticmethod
    def get_neural_model_inputs():
        return {"states_embedding": torch.zeros(2, 4, 4)}


def test_checkpoint_reconstruction_instantiates_enabled_force_head():
    model = _model(force_head=True, num_bodies=5)
    force_rms = RunningMeanStd(shape=(5, 3), device="cpu")
    force_rms.update(torch.arange(30, dtype=torch.float32).reshape(2, 5, 3), batch_dim=True)
    model.set_body_net_contact_force_rms(force_rms)
    cfg = {
        "inputs": {"low_dim": ["states_embedding"]},
        "network": deepcopy(_network_cfg(force_head=True)),
    }
    checkpoint = {
        "version": 2,
        "cfg": cfg,
        "rms_info": model.get_rms_info(),
        "model_state_dict": model.state_dict(),
    }

    restored = reconstruct_model_from_checkpoint(checkpoint, _CheckpointSolver(), device="cpu")
    output = restored.forward_with_aux({"states_embedding": torch.randn(2, 4, 4)}, single_step=True)

    assert restored.predict_body_net_contact_forces
    torch.testing.assert_close(restored.body_net_contact_force_rms.mean, force_rms.mean)
    torch.testing.assert_close(restored.body_net_contact_force_rms.var, force_rms.var)
    assert output.state_prediction.shape == (2, 1, 9)
    assert output.auxiliary_predictions["body_net_contact_forces_b"].shape == (2, 1, 5, 3)


class _AuxiliaryModel(nn.Module):
    predict_body_net_contact_forces = True
    num_bodies = 4

    def forward_with_aux(self, _inputs, single_step=False):
        assert single_step
        forces = torch.arange(24, dtype=torch.float32).reshape(2, 1, 4, 3)
        return ModelMixedInputOutput(
            state_prediction=torch.ones(2, 1, 9),
            auxiliary_predictions={"body_net_contact_forces_b": forces},
        )


def test_solver_caches_latest_force_prediction_without_changing_state_prediction():
    solver = NeuralSolver.__new__(NeuralSolver)
    solver.neural_model = _AuxiliaryModel()
    solver.num_envs = 2
    solver.num_contact_bodies_per_env = 4
    solver.torch_device = torch.device("cpu")

    NeuralSolver._configure_auxiliary_prediction_cache(solver)
    state_prediction = NeuralSolver._forward_neural_model(solver, {})

    assert state_prediction.shape == (2, 1, 9)
    torch.testing.assert_close(
        solver.predicted_body_net_contact_forces_b,
        torch.arange(24, dtype=torch.float32).reshape(2, 4, 3),
    )


def test_solver_reset_clears_cached_force_predictions():
    solver = NeuralSolver.__new__(NeuralSolver)
    solver.predicted_body_net_contact_forces_b = torch.ones(3, 4, 3)
    solver.torch_device = "cpu"

    NeuralSolver.reset(solver, env_ids=torch.tensor([1]))
    assert torch.count_nonzero(solver.predicted_body_net_contact_forces_b[1]) == 0
    assert torch.count_nonzero(solver.predicted_body_net_contact_forces_b[[0, 2]]) > 0

    NeuralSolver.reset(solver)
    assert torch.count_nonzero(solver.predicted_body_net_contact_forces_b) == 0


def test_solver_reset_without_force_prediction_cache():
    solver = NeuralSolver.__new__(NeuralSolver)
    solver.predicted_body_net_contact_forces_b = None

    NeuralSolver.reset(solver, env_ids=torch.tensor([0]))
    NeuralSolver.reset(solver)


def test_world_force_labels_rotate_into_current_root_axes():
    angle = torch.tensor(torch.pi / 4)
    root_body_q = torch.zeros(1, 2, 7)
    root_body_q[..., 5] = angle.sin()
    root_body_q[..., 6] = angle.cos()
    forces_w = torch.tensor([[[[2.0, 0.0, 0.0]], [[0.0, 3.0, 0.0]]]])

    forces_b = transform_body_net_contact_forces_to_current_root_frame(forces_w, root_body_q)

    torch.testing.assert_close(
        forces_b,
        torch.tensor([[[[0.0, -2.0, 0.0]], [[3.0, 0.0, 0.0]]]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_enabled_head_requires_force_labels_in_all_datasets():
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.force_head_enabled = True
    trainer.train_dataset = SimpleNamespace(data_keys=["states", BODY_FORCE_LABEL_KEY])
    trainer.valid_datasets = {"missing": SimpleNamespace(data_keys=["states"])}

    with pytest.raises(ValueError, match=BODY_FORCE_LABEL_KEY):
        trainer._validate_force_label_datasets()


def test_enabled_head_requires_consistent_force_body_order():
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.force_head_enabled = True
    trainer.train_dataset = SimpleNamespace(
        data_keys=["states", BODY_FORCE_LABEL_KEY],
        body_net_contact_force_body_names=("base", "LF_FOOT"),
    )
    trainer.valid_datasets = {
        "reversed": SimpleNamespace(
            data_keys=["states", BODY_FORCE_LABEL_KEY],
            body_net_contact_force_body_names=("LF_FOOT", "base"),
        )
    }

    with pytest.raises(ValueError, match="body-name order"):
        trainer._validate_force_label_datasets()


class _LossSolver:
    dof_q_per_env = 4

    @staticmethod
    def convert_prediction_to_next_states(*, states, prediction, dt):
        del states, dt
        return prediction

    @staticmethod
    def wrap2PI(_states):
        return None


def test_normalized_force_loss_has_metrics_and_head_gradients():
    model = _model(force_head=True, num_bodies=7)
    force_rms = RunningMeanStd(shape=(7, 3), device="cpu")
    force_rms.update(torch.randn(16, 7, 3) * 20.0 + 5.0, batch_dim=True)
    model.set_body_net_contact_force_rms(force_rms)

    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.neural_model = model
    trainer.force_head_enabled = True
    trainer.force_loss_weight = 0.25
    trainer.active_contact_force_threshold_n = 1.0
    trainer.loss_weights = torch.ones(9)
    trainer.loss_func = nn.MSELoss()
    trainer.batch_size = 2
    trainer.neural_solver = _LossSolver()
    trainer.neural_env = SimpleNamespace(frame_dt=0.01)
    data = {
        "states_embedding": torch.randn(2, 4, 4),
        "states": torch.zeros(2, 4, 9),
        "next_states": torch.zeros(2, 4, 9),
        "prediction_target": torch.zeros(2, 4, 9),
        BODY_FORCE_TARGET_KEY: torch.randn(2, 4, 7, 3) * 10.0,
    }

    loss, metrics = trainer.compute_loss(data, train=True)
    loss.backward()

    assert metrics["body_force_normalized_MSE"] > 0.0
    assert metrics["body_force_RMSE_N"] > 0.0
    assert metrics["body_force_active_RMSE_N"] > 0.0
    assert metrics["body_force_support_consistency_RMSE_N"] > 0.0
    assert model.body_net_contact_force_head.output_net.weight.grad is not None
    assert torch.count_nonzero(model.transformer_model.transformer.wte.weight.grad) > 0


def test_force_metrics_use_vector_rmse_and_active_threshold():
    target = torch.tensor([[[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]])
    prediction = torch.tensor([[[[0.0, 4.0, 0.0], [0.0, 0.0, 0.0]]]])

    metrics = body_net_contact_force_metrics(prediction, target, active_contact_threshold_n=1.0)

    assert metrics["body_force_RMSE_N"] == pytest.approx((25.0 / 2.0) ** 0.5)
    assert metrics["body_force_contact_sensor_reconstruction_RMSE_N"] == metrics["body_force_RMSE_N"]
    assert metrics["body_force_active_RMSE_N"] == pytest.approx(3.0)
    assert metrics["body_force_active_fraction"] == pytest.approx(0.5)
    assert metrics["body_force_RMSE_body_00_N"] == pytest.approx(4.0)
    assert metrics["body_force_RMSE_body_01_N"] == pytest.approx(3.0)


def test_force_rms_serialization_round_trips_for_ddp_broadcast():
    source = VanillaTrainer.__new__(VanillaTrainer)
    source.dataset_rms = {
        BODY_FORCE_TARGET_KEY: RunningMeanStd(shape=(4, 3), device="cpu"),
        "prediction_target": RunningMeanStd(shape=(9,), device="cpu"),
    }
    source.dataset_rms[BODY_FORCE_TARGET_KEY].update(
        torch.arange(60, dtype=torch.float32).reshape(5, 4, 3),
        batch_dim=True,
    )
    state = source._serialize_dataset_rms()

    destination = VanillaTrainer.__new__(VanillaTrainer)
    destination.device = "cpu"
    destination._load_dataset_rms_state(state)

    actual = destination.dataset_rms[BODY_FORCE_TARGET_KEY]
    expected = source.dataset_rms[BODY_FORCE_TARGET_KEY]
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(actual.var, expected.var)
    torch.testing.assert_close(actual.count, expected.count)


def test_old_state_only_v2_checkpoint_reconstructs_strictly():
    old_model = _model()
    checkpoint = {
        "version": 2,
        "cfg": {
            "inputs": {"low_dim": ["states_embedding"]},
            "network": _network_cfg(),
        },
        "rms_info": {},
        "model_state_dict": old_model.state_dict(),
    }

    restored = reconstruct_model_from_checkpoint(checkpoint, _CheckpointSolver(), device="cpu")

    assert not restored.predict_body_net_contact_forces
    assert restored.body_net_contact_force_rms is None
    restored.load_state_dict(old_model.state_dict(), strict=True)
