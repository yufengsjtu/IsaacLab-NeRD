# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO101 robot normal-force training and checkpoint contracts."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import pytest
import torch
from isaaclab_neural.contacts.keyboard_normal_force import (
    BODY_NAMES,
    LABEL_KEY,
    MODEL_CONTRACT,
    normal_force_loss,
    publish_normal_forces,
    validate_force_metadata,
)
from isaaclab_neural.models.models import ModelMixedInput
from isaaclab_neural.utils.checkpoint import reconstruct_model_from_checkpoint, save_checkpoint
from isaaclab_neural.utils.running_mean_std import RunningMeanStd


def model_config():
    return {
        "normalize_input": False,
        "normalize_output": True,
        "encoder": {"low_dim": {"layer_sizes": [], "activation": "relu", "layernorm": False}},
        "model": {"output_tanh": False, "mlp": {"layer_sizes": [16], "activation": "relu", "layernorm": False}},
        "robot_net_normal_force_head": {"enabled": True},
    }


def make_model():
    inputs = {"states_embedding": torch.zeros(2, 3, 5)}
    cfg = model_config()
    model = ModelMixedInput(inputs, 5, {"low_dim": ["states_embedding"]}, cfg, device="cpu", num_bodies=116)
    model.set_output_rms(RunningMeanStd(shape=(5,), device="cpu"))
    model.set_robot_net_normal_force_rms(RunningMeanStd(shape=(7, 3), device="cpu"))
    return model, inputs, cfg


def test_normal_force_head_has_independent_contract_and_finite_gradients():
    model, inputs, _ = make_model()
    output = model(inputs, return_aux=True)
    assert output.auxiliary_predictions[LABEL_KEY].shape == (2, 3, 7, 3)
    target = torch.zeros(2, 3, 7, 3)
    target[:, :, -1, 2] = 5.0
    loss, metrics = normal_force_loss(output.auxiliary_predictions, target, model.robot_net_normal_force_rms)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["robot_normal_active_fraction"] > 0
    assert any(parameter.grad is not None for parameter in model.robot_net_normal_force_head.parameters())
    assert model.robot_normal_force_contract == MODEL_CONTRACT


def test_unique_supervision_masks_state_force_bce_and_metrics():
    from isaaclab_neural.train.trainers import VanillaTrainer

    model, inputs, _ = make_model()
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.neural_model = model
    trainer.force_head_enabled = False
    trainer.robot_force_head_enabled = True
    trainer.robot_force_loss_weight = 1.0
    trainer.loss_weights = torch.ones(5)
    trainer.loss_func = torch.nn.MSELoss()
    trainer.neural_env = SimpleNamespace(frame_dt=0.01)
    trainer.neural_solver = SimpleNamespace(
        dof_q_per_env=2, convert_prediction_to_next_states=lambda **kwargs: kwargs["prediction"], wrap2PI=lambda _: None
    )
    trainer.cfg = {"keyboard_dataset_contracts": {"train": {"num_q": 2, "key_q_indices": [1]}}}
    mask = torch.tensor([[True, False, True], [False, True, False]])
    data = {
        **inputs,
        "states": torch.zeros(2, 3, 5),
        "next_states": torch.zeros(2, 3, 5),
        "prediction_target": torch.zeros(2, 3, 5, requires_grad=True),
        LABEL_KEY: torch.zeros(2, 3, 7, 3, requires_grad=True),
        "supervision_mask": mask,
    }
    loss, metrics = trainer.compute_loss(data, train=True)
    changed = {key: value.detach().clone() for key, value in data.items()}
    for key in ("next_states", "prediction_target", LABEL_KEY):
        changed[key][~mask] = 10000
    changed_loss, changed_metrics = trainer.compute_loss(changed, train=True)
    torch.testing.assert_close(loss, changed_loss)
    assert metrics == changed_metrics
    loss.backward()
    assert torch.count_nonzero(data["prediction_target"].grad[~mask]) == 0
    assert torch.count_nonzero(data[LABEL_KEY].grad[~mask]) == 0
    assert any(parameter.grad is not None for parameter in model.robot_net_normal_force_head.parameters())


def test_normal_checkpoint_roundtrip_rejects_changed_semantics(tmp_path):
    model, inputs, network = make_model()
    cfg = {"inputs": {"low_dim": ["states_embedding"]}, "network": network}
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, "SO101-Keyboard-Fixed", cfg)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    solver = SimpleNamespace(
        get_neural_model_inputs=lambda: inputs,
        prediction_dim=5,
        contact_mode="newton_native",
        num_contact_bodies_per_env=116,
    )
    restored = reconstruct_model_from_checkpoint(checkpoint, solver, device="cpu")
    torch.testing.assert_close(
        model(inputs, return_aux=True).auxiliary_predictions[LABEL_KEY],
        restored(inputs, return_aux=True).auxiliary_predictions[LABEL_KEY],
    )
    checkpoint["robot_normal_force_contract"]["frame"] = "body"
    with pytest.raises(ValueError, match="normal-force contract"):
        reconstruct_model_from_checkpoint(checkpoint, solver, device="cpu")


def test_robot_force_cache_reset_is_selective_and_cached_inference_is_rejected():
    from isaaclab_neural.solvers.neural_solver import NeuralSolver

    solver = object.__new__(NeuralSolver)
    solver.num_envs = 3
    solver.torch_device = torch.device("cpu")
    setattr(solver, "use_kv_cache", False)
    model, _, _ = make_model()
    solver.set_neural_solver_model(model)
    assert solver.predicted_robot_net_normal_forces_w is not None
    solver.predicted_robot_net_normal_forces_w.fill_(1.0)
    solver.reset(torch.tensor([1]))
    torch.testing.assert_close(solver.predicted_robot_net_normal_forces_w[1], torch.zeros(7, 3))
    torch.testing.assert_close(solver.predicted_robot_net_normal_forces_w[0], torch.ones(7, 3))
    setattr(solver, "use_kv_cache", True)
    with pytest.raises(ValueError, match="uncached"):
        solver.set_neural_solver_model(model)


@pytest.mark.parametrize("load_mode", ["eager", "lazy"])
def test_keyboard_loader_keeps_structured_normal_force_labels(tmp_path, load_mode):
    from isaaclab_neural.data import create_trajectory_dataset
    from isaaclab_neural.data.keyboard import write_keyboard_dataset
    from test_keyboard_dataset import sample

    rollout, metadata = sample()
    metadata["force_body_names"] = list(BODY_NAMES)
    path = tmp_path / "keyboard.hdf5"
    write_keyboard_dataset(path, rollout, metadata)
    dataset = create_trajectory_dataset(
        load_mode=load_mode,
        hdf5_dataset_path=path,
        sample_sequence_length=4,
        contact_representation="contact_tokens",
        max_contact_tokens=8,
        contact_packing_policy="body_round_robin_pair_atomic",
    )
    assert dataset[0][LABEL_KEY].shape == (4, 7, 3)


def test_sensor_publication_uses_normal_minus_zero_friction(monkeypatch):
    import warp as wp

    total = torch.zeros(14, 3)
    friction = torch.ones_like(total)
    sensor = SimpleNamespace(
        sensor_names=list(BODY_NAMES),
        cfg=SimpleNamespace(filter_prim_paths_expr=[]),
        _newton_total_force_view=total,
        _newton_total_force_friction_view=friction,
    )
    scene = SimpleNamespace(num_envs=2, sensors={"robot_contact": sensor})
    monkeypatch.setattr(wp, "to_torch", lambda array: array)
    forces = torch.arange(42, dtype=torch.float32).reshape(2, 7, 3)
    publish_normal_forces(scene, forces)
    torch.testing.assert_close((total - friction).reshape_as(forces), forces)
    assert not friction.any()
    sensor.sensor_names.reverse()
    with pytest.raises(ValueError, match="body order"):
        publish_normal_forces(scene, forces)


def test_keyboard_position_loss_scale_changes_only_key_position_coordinates():
    from isaaclab_neural.train.trainers import keyboard_position_loss_weights

    weights = torch.arange(1, 9, dtype=torch.float32)
    scaled = keyboard_position_loss_weights(weights, num_q=4, key_q_indices=[2, 3], scale_m=0.0001)
    torch.testing.assert_close(scaled[[2, 3]], torch.full((2,), 10000.0))
    torch.testing.assert_close(scaled[[0, 1, 4, 5, 6, 7]], weights[[0, 1, 4, 5, 6, 7]])
    torch.testing.assert_close(weights, torch.arange(1, 9, dtype=torch.float32))
    balanced = keyboard_position_loss_weights(
        weights, num_q=4, key_q_indices=[2, 3], scale_m=0.0001, robot_scale_rad=0.001
    )
    torch.testing.assert_close(balanced[:2], torch.full((2,), 1000.0))
    torch.testing.assert_close(balanced[2:4], torch.full((2,), 10000.0))
    torch.testing.assert_close(balanced[4:], weights[4:])
    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="scale"):
            keyboard_position_loss_weights(weights, num_q=4, key_q_indices=[2, 3], scale_m=scale)
        with pytest.raises(ValueError, match="scale"):
            keyboard_position_loss_weights(
                weights, num_q=4, key_q_indices=[2, 3], scale_m=0.0001, robot_scale_rad=scale
            )
    with pytest.raises(ValueError, match="indices"):
        keyboard_position_loss_weights(weights, num_q=4, key_q_indices=[4], scale_m=0.0001)


def test_training_adapter_honors_physics_frame_override_without_changing_legacy_default():
    from isaaclab_neural.train.trainers import TrainingEnvAdapter

    env = SimpleNamespace(step_dt=0.04, physics_dt=0.01, neural_adapter=SimpleNamespace())
    assert TrainingEnvAdapter(env).frame_dt == 0.04
    env.frame_dt = 0.01
    assert TrainingEnvAdapter(env).frame_dt == 0.01


def test_keyboard_neural_config_is_copyable_and_keeps_native_control_contract():
    from isaaclab_neural.envs.keyboard_nerd_env import NerdSO101KeyboardEnvCfg
    from isaaclab_neural.physics import NerdNewtonCfg

    cfg = NerdSO101KeyboardEnvCfg()
    copied = cfg.copy()
    assert isinstance(copied.sim.physics, NerdNewtonCfg)
    assert copied.sim.dt == 0.01 and copied.decimation == 4
    assert copied.sim.physics.num_substeps == 1
    assert len(copied.scene.keyboard.spawn.assets_cfg) == 1
    assert copied.scene.keyboard.spawn.assets_cfg[0].seed == 0
    assert copied.robot_normal_force_sensor == "robot_contact"
    for event in (copied.events.reset_keyboard, copied.commands.typing.reset.pre_solve_reset):
        assert all(tuple(bounds) == (0, 0) for bounds in event.params["pose_range"].values())
    copied.scene.num_envs = 2
    assert cfg.scene.num_envs == 8


def test_current_usd_body_aliases_preserve_force_order():
    from isaaclab_neural.contacts.keyboard_normal_force import BODY_NAMES, validate_sensor

    sensor = SimpleNamespace(
        sensor_names=[name + "_link" for name in BODY_NAMES], cfg=SimpleNamespace(filter_prim_paths_expr=[])
    )
    scene = SimpleNamespace(sensors={"robot_contact": sensor})
    validate_sensor(scene)
    sensor.sensor_names.reverse()
    with pytest.raises(ValueError, match="body order"):
        validate_sensor(scene)


def test_force_dataset_requires_keyboard_metadata(tmp_path):
    with h5py.File(tmp_path / "invalid.hdf5", "w") as file:
        data = file.create_group("data")
        data.create_dataset("states", data=torch.zeros(1, 2, 228).numpy())
        data.create_dataset(LABEL_KEY, data=torch.zeros(1, 2, 7, 3).numpy())
        with pytest.raises(ValueError, match="keyboard"):
            validate_force_metadata(data)
