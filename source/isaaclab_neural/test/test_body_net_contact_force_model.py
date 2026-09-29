# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the optional per-body resultant-contact-force model output."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch
import torch.nn as nn
from isaaclab_neural.data import BatchTransitionDataset
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


class _NeuralTestHarness(NeuralSolver):
    predict = NeuralSolver._forward_neural_model


class _TrainerTestHarness(VanillaTrainer):
    validate_force_labels = VanillaTrainer._validate_force_label_datasets
    serialize_dataset_rms = VanillaTrainer._serialize_dataset_rms
    load_dataset_rms_state = VanillaTrainer._load_dataset_rms_state


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

    assert model.transformer_model is not None
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
    trunk_weight = model.get_parameter("transformer_model.transformer.wte.weight")
    assert trunk_weight.grad is not None
    assert torch.count_nonzero(trunk_weight.grad) > 0


def test_filtered_finger_normal_force_head_roundtrip_and_gradients():
    cfg = _network_cfg(force_head=True)
    cfg["finger_object_normal_force_head"] = {"enabled": True}
    model = ModelMixedInput(
        input_sample={"states_embedding": torch.zeros(2, 4, 4)},
        output_dim=9,
        input_cfg={"low_dim": ["states_embedding"]},
        network_cfg=cfg,
        num_bodies=5,
        device="cpu",
    )
    inputs = {"states_embedding": torch.randn(2, 4, 4)}
    output = model.forward_with_aux(inputs)
    force = output.auxiliary_predictions["finger_object_normal_forces_b"]
    assert force.shape == (2, 4, 2, 3)
    assert output.auxiliary_predictions["body_net_contact_forces_b"].shape == (2, 4, 5, 3)
    (
        output.auxiliary_predictions["finger_object_normal_forces_raw_b"].square().mean()
        + output.auxiliary_predictions["finger_object_contact_logits"].square().mean()
    ).backward()
    assert model.finger_object_contact_head.output_net.weight.grad is not None
    assert model.finger_object_normal_force_head.output_net.weight.grad is not None
    assert model.get_parameter("transformer_model.transformer.wte.weight").grad is not None
    rms = RunningMeanStd(shape=(2, 3), device="cpu")
    rms.update(torch.randn(2, 4, 2, 3), batch_dim=True, time_dim=True)
    model.set_finger_object_normal_force_rms(rms)
    model.normalize_output = True
    cfg["normalize_output"] = True
    model.set_output_rms(RunningMeanStd(shape=(9,), device="cpu"))
    model.set_body_net_contact_force_rms(RunningMeanStd(shape=(5, 3), device="cpu"))
    checkpoint = {
        "version": 2,
        "cfg": {"inputs": {"low_dim": ["states_embedding"]}, "network": cfg},
        "rms_info": model.get_rms_info(),
        "model_state_dict": model.state_dict(),
        "finger_object_force_contract": model.finger_object_force_contract,
    }
    broken = dict(checkpoint, finger_object_force_contract={})
    with pytest.raises(ValueError, match="contract"):
        reconstruct_model_from_checkpoint(broken, _CheckpointSolver(), device="cpu")
    restored = reconstruct_model_from_checkpoint(checkpoint, _CheckpointSolver(), device="cpu")
    model.eval()
    expected = model.forward_with_aux(inputs, single_step=True)
    actual = restored.forward_with_aux(inputs, single_step=True)
    torch.testing.assert_close(
        actual.auxiliary_predictions["finger_object_normal_forces_b"],
        expected.auxiliary_predictions["finger_object_normal_forces_b"],
    )
    with torch.no_grad():
        restored.finger_object_contact_head.output_net.weight.zero_()
        restored.finger_object_contact_head.output_net.bias.fill_(-100.0)
    inactive = restored.forward_with_aux(inputs).auxiliary_predictions["finger_object_normal_forces_b"]
    assert torch.count_nonzero(inactive) == 0


@pytest.mark.parametrize("cache_flag", ["use_kv_cache", "use_feature_cache"])
def test_filtered_force_head_rejects_state_only_cached_inference(cache_flag):
    solver = _NeuralTestHarness.__new__(_NeuralTestHarness)
    solver.num_envs = 2
    solver.torch_device = "cpu"
    setattr(solver, cache_flag, True)
    model = torch.nn.Identity()
    model.predict_finger_object_normal_forces = True
    with pytest.raises(ValueError, match="uncached auxiliary"):
        solver.set_neural_solver_model(model)


def test_resume_does_not_silently_disable_requested_finger_head(tmp_path, monkeypatch):
    import sys

    import yaml

    monkeypatch.setattr(sys, "argv", ["train.py"])
    from isaaclab_neural.train import train

    cfg = {
        "env": {"env_name": "Franka-Lift", "neural_solver_cfg": {}},
        "inputs": {},
        "algorithm": {},
        "network": {"finger_object_normal_force_head": {"enabled": True}},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    checkpoint_cfg = deepcopy(cfg)
    checkpoint_cfg["network"] = {}
    monkeypatch.setattr(train, "load_checkpoint", lambda *args, **kwargs: {"cfg": checkpoint_cfg})
    args = SimpleNamespace(
        cfg=str(path),
        checkpoint="state_only.pt",
        device="cpu",
        cfg_overrides="",
        no_time_stamp=True,
        num_envs=None,
        render=False,
        seed=0,
        update_dataset_statistics=False,
        test=False,
    )
    with pytest.raises(ValueError, match="finger.*head"):
        train.load_training_cfg(args)


@pytest.mark.parametrize(
    ("checkpoint_solver_cfg", "overrides", "compatible"),
    [
        ({"contact_token_frame": "owner"}, "env.neural_solver_cfg.use_kv_cache True", True),
        ({"contact_token_frame": "owner"}, "env.neural_solver_cfg.contact_token_frame root", False),
        ({}, "", False),
        ({"contact_token_frame": "owner"}, "inputs.contact_set.dim 18", False),
        ({"contact_token_frame": "root"}, "", False),
    ],
)
def test_resume_rejects_changed_input_semantics(tmp_path, monkeypatch, checkpoint_solver_cfg, overrides, compatible):
    import sys

    import yaml

    monkeypatch.setattr(sys, "argv", ["train.py"])
    from isaaclab_neural.train import train

    cfg = {
        "env": {
            "env_name": "SO101-Keyboard-Fixed",
            "neural_solver_cfg": {"contact_token_frame": "owner", "use_kv_cache": False},
        },
        "inputs": {"low_dim": ["states_embedding"], "contact_set": {"dim": 17}},
        "algorithm": {},
        "network": {},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    checkpoint_cfg = deepcopy(cfg)
    checkpoint_cfg["env"]["neural_solver_cfg"] = checkpoint_solver_cfg
    monkeypatch.setattr(train, "load_checkpoint", lambda *args, **kwargs: {"cfg": checkpoint_cfg})
    args = SimpleNamespace(
        cfg=str(path),
        checkpoint="owner.pt",
        device="cpu",
        cfg_overrides=overrides,
        no_time_stamp=True,
        num_envs=None,
        render=False,
        seed=0,
        update_dataset_statistics=True,
        test=False,
    )
    if compatible:
        train.load_training_cfg(args)
    else:
        with pytest.raises(ValueError, match="input semantics"):
            train.load_training_cfg(args)


@pytest.mark.parametrize(
    ("checkpoint_contract", "compatible"),
    [
        ({"unit": "unique_transition", "window_sampling": "sliding"}, True),
        ({"unit": "unique_transition", "window_sampling": "canonical"}, False),
        (None, False),
    ],
)
def test_reused_checkpoint_statistics_require_matching_sampling_contract(checkpoint_contract, compatible):
    def contract(value):
        if value is None:
            return None
        sliding = value["window_sampling"] == "sliding"
        return {
            **value,
            "scope": "global",
            "supervision": "sliding_multi_position_v1" if sliding else "canonical_disjoint_blocks_v1",
            "normalization_sampling": "training_windows" if sliding else "unique_targets",
            "train": {"train": {"unique_selected": 8}},
        }

    trainer = _TrainerTestHarness.__new__(_TrainerTestHarness)
    trainer.cfg = {"dataset_capacity_contract": contract({"unit": "unique_transition", "window_sampling": "sliding"})}
    trainer._checkpoint = {"cfg": {"dataset_capacity_contract": contract(checkpoint_contract)}}
    if compatible:
        trainer._validate_checkpoint_normalization_contract()
    else:
        with pytest.raises(ValueError, match="update-dataset-statistics"):
            trainer._validate_checkpoint_normalization_contract()


def test_filtered_force_training_rejects_state_only_datasets():
    trainer = _TrainerTestHarness.__new__(_TrainerTestHarness)
    trainer.force_head_enabled = False
    trainer.finger_force_head_enabled = True
    trainer.train_dataset = SimpleNamespace(data_keys=["states", "next_states", "joint_f"])
    trainer.valid_datasets = {}
    with pytest.raises(ValueError, match="finger_object_normal_forces_w"):
        trainer.validate_force_labels()


@pytest.mark.parametrize("force_heads", [False, True])
def test_model_device_transfer_includes_normalizers(force_heads):
    cfg = _network_cfg(force_head=force_heads)
    cfg["finger_object_normal_force_head"] = {"enabled": force_heads}
    model = ModelMixedInput(
        input_sample={"states_embedding": torch.zeros(2, 4, 4)},
        output_dim=9,
        input_cfg={"low_dim": ["states_embedding"]},
        network_cfg=cfg,
        num_bodies=5,
        device="cpu",
    )
    model.set_input_rms({"states_embedding": RunningMeanStd(shape=(4,), device="cpu")})
    model.set_output_rms(RunningMeanStd(shape=(9,), device="cpu"))
    if force_heads:
        model.set_body_net_contact_force_rms(RunningMeanStd(shape=(5, 3), device="cpu"))
        model.set_finger_object_normal_force_rms(RunningMeanStd(shape=(2, 3), device="cpu"))
    result = model.to("meta")
    assert all(value.device.type == "meta" for value in model.state_dict().values())
    assert model.device == "meta"
    assert result is model


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
    assert restored.body_net_contact_force_rms is not None
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
    solver = _NeuralTestHarness.__new__(_NeuralTestHarness)
    solver.num_envs = 2
    solver.num_contact_bodies_per_env = 4
    solver.torch_device = "cpu"

    solver.set_neural_solver_model(_AuxiliaryModel())
    state_prediction = solver.predict({})

    assert state_prediction.shape == (2, 1, 9)
    torch.testing.assert_close(
        solver.predicted_body_net_contact_forces_b,
        torch.arange(24, dtype=torch.float32).reshape(2, 4, 3),
    )


def test_solver_rejects_noncallable_auxiliary_forward(monkeypatch):
    solver = _NeuralTestHarness.__new__(_NeuralTestHarness)
    solver.neural_model = _AuxiliaryModel()
    monkeypatch.setattr(solver.neural_model, "forward_with_aux", torch.tensor(1.0))
    solver.predicted_body_net_contact_forces_b = torch.zeros(2, 4, 3)
    with pytest.raises(TypeError, match="callable forward_with_aux"):
        solver.predict({})


def test_solver_reset_clears_cached_force_predictions():
    solver = _NeuralTestHarness.__new__(_NeuralTestHarness)
    solver.predicted_body_net_contact_forces_b = torch.ones(3, 4, 3)
    solver.torch_device = "cpu"

    NeuralSolver.reset(solver, env_ids=torch.tensor([1]))
    assert torch.count_nonzero(solver.predicted_body_net_contact_forces_b[1]) == 0
    assert torch.count_nonzero(solver.predicted_body_net_contact_forces_b[[0, 2]]) > 0

    NeuralSolver.reset(solver)
    assert torch.count_nonzero(solver.predicted_body_net_contact_forces_b) == 0


def test_solver_reset_without_force_prediction_cache():
    solver = _NeuralTestHarness.__new__(_NeuralTestHarness)
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
    trainer = _TrainerTestHarness.__new__(_TrainerTestHarness)
    trainer.force_head_enabled = True
    trainer.train_dataset = create_autospec(
        BatchTransitionDataset, instance=True, data_keys=["states", BODY_FORCE_LABEL_KEY]
    )
    trainer.valid_datasets = {"missing": SimpleNamespace(data_keys=["states"])}

    with pytest.raises(ValueError, match=BODY_FORCE_LABEL_KEY):
        trainer.validate_force_labels()


def test_enabled_head_requires_consistent_force_body_order():
    trainer = _TrainerTestHarness.__new__(_TrainerTestHarness)
    trainer.force_head_enabled = True
    trainer.train_dataset = create_autospec(
        BatchTransitionDataset,
        instance=True,
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
        trainer.validate_force_labels()


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

    trainer = _TrainerTestHarness.__new__(_TrainerTestHarness)
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
    trunk_weight = model.get_parameter("transformer_model.transformer.wte.weight")
    assert trunk_weight.grad is not None
    assert torch.count_nonzero(trunk_weight.grad) > 0


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
    source = _TrainerTestHarness.__new__(_TrainerTestHarness)
    source.dataset_rms = {
        BODY_FORCE_TARGET_KEY: RunningMeanStd(shape=(4, 3), device="cpu"),
        "prediction_target": RunningMeanStd(shape=(9,), device="cpu"),
    }
    source.dataset_rms[BODY_FORCE_TARGET_KEY].update(
        torch.arange(60, dtype=torch.float32).reshape(5, 4, 3),
        batch_dim=True,
    )
    state = source.serialize_dataset_rms()

    destination = _TrainerTestHarness.__new__(_TrainerTestHarness)
    destination.device = "cpu"
    destination.load_dataset_rms_state(state)

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
