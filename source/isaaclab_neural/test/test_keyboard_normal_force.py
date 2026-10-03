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


KEY_HEAD = {
    "enabled": True,
    "key_q_start": 2,
    "num_keys": 3,
    "key_body_slot_start": 3,
    "history_steps": 2,
    "embed_dim": 4,
    "summary_dim": 8,
    "context_dim": 8,
    "hidden_sizes": [16],
}


def make_key_head_model(transformer_overrides: dict | None = None, joint_target: bool = False):
    """Two robot joints (q 0-1) and three keys (q 2-4); key i owns contact body slot 3 + i of six."""
    from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_BODY_SLOT_INDEX, CONTACT_TOKEN_VALID_INDEX

    torch.manual_seed(0)
    tokens = torch.randn(2, 4, 3, 17)
    tokens[..., CONTACT_TOKEN_VALID_INDEX] = torch.tensor([1.0, 1.0, 0.0])
    tokens[..., CONTACT_TOKEN_BODY_SLOT_INDEX] = torch.tensor([4.0, 1.0, 0.0])
    inputs = {
        "states_embedding": torch.randn(2, 4, 10),
        "joint_f": torch.randn(2, 4, 5),
        "gravity_dir": torch.randn(2, 4, 3),
        "contact_tokens": tokens,
    }
    if joint_target:
        inputs["joint_target_offset"] = torch.randn(2, 4, 5)
    network = {
        **model_config(),
        "normalize_input": True,
        "key_head": dict(KEY_HEAD),
        "transformer": {
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
            "block_size": 8,
            "dropout": 0.0,
            "bias": False,
            **(transformer_overrides or {}),
        },
    }
    input_cfg = {
        "low_dim": ["states_embedding", "joint_f", "gravity_dir", *(["joint_target_offset"] if joint_target else [])],
        "contact_set": {"dim": 17, "encoder_type": "shared_per_body", "body_latent_dim": 4, "hidden_dim": 8},
    }
    model = ModelMixedInput(inputs, 10, input_cfg, network, device="cpu", num_bodies=6)
    rms = {}
    for name, shape in (
        ("states_embedding", (10,)),
        ("joint_f", (5,)),
        ("gravity_dir", (3,)),
        ("contact_tokens", (17,)),
        *([("joint_target_offset", (5,))] if joint_target else []),
    ):
        rms[name] = RunningMeanStd(shape=shape, device="cpu")
        rms[name].update(torch.randn(64, *shape) * torch.rand(shape) + torch.randn(shape), batch_dim=True)
    model.set_input_rms(rms)
    output_rms = RunningMeanStd(shape=(10,), device="cpu")
    output_rms.update(torch.randn(64, 10) * torch.rand(10) + torch.randn(10), batch_dim=True)
    model.set_output_rms(output_rms)
    model.set_robot_net_normal_force_rms(RunningMeanStd(shape=(7, 3), device="cpu"))
    return model, inputs, input_cfg, network


def test_key_head_assembles_robot_and_pooled_key_outputs_in_original_order():
    model, inputs, _, _ = make_key_head_model()
    output = model({k: v.clone() for k, v in inputs.items()}, return_aux=True)
    assert output.state_prediction.shape == (2, 4, 10) and bool(torch.isfinite(output.state_prediction).all())
    output.state_prediction.square().sum().backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    unused_heads = ("robot_net_normal_force_head", "robot_normal_contact_head")
    assert [name for name in missing if not name.startswith(unused_heads)] == []

    # Zero decoder outputs must map to per-coordinate robot means and pooled key means.
    with torch.no_grad():
        for layer in (model.model.output_net, model.key_head.decoder[-1]):
            layer.weight.zero_()
            layer.bias.zero_()
        state = model({k: v.clone() for k, v in inputs.items()})
    mean = model.output_rms.mean
    robot = [0, 1, 5, 6]
    torch.testing.assert_close(state[..., robot], mean[robot].expand_as(state[..., robot]))
    for keys in ([2, 3, 4], [7, 8, 9]):
        torch.testing.assert_close(state[..., keys], mean[keys].mean().expand_as(state[..., keys]))


def test_key_head_uses_robot_joint_target_offsets_only():
    model, inputs, _, _ = make_key_head_model(joint_target=True)
    baseline, _, _, _ = make_key_head_model()
    assert model.key_head.trunk_input_dim == baseline.key_head.trunk_input_dim + 2
    with torch.no_grad():
        reference = model({k: v.clone() for k, v in inputs.items()})
        changed = {k: v.clone() for k, v in inputs.items()}
        changed["joint_target_offset"][..., 2:] += 1.0
        torch.testing.assert_close(model(changed), reference)
        changed["joint_target_offset"][..., 0] += 1.0
        assert not torch.allclose(model(changed), reference)
    with pytest.raises(ValueError, match="optionally followed"):
        bad_inputs = {**inputs, "joint_target_q": inputs["joint_target_offset"]}
        bad_cfg = {
            "low_dim": ["states_embedding", "joint_f", "gravity_dir", "joint_target_q"],
            "contact_set": {"dim": 17, "encoder_type": "shared_per_body", "body_latent_dim": 4, "hidden_dim": 8},
        }
        _, _, _, network = make_key_head_model()
        ModelMixedInput(bad_inputs, 10, bad_cfg, network, device="cpu", num_bodies=6)


@pytest.mark.parametrize("joint_target", [False, True])
@pytest.mark.parametrize("steps", [1, 2, 4])
def test_key_head_single_step_and_encoded_paths_match_the_full_window(steps, joint_target):
    model, inputs, _, _ = make_key_head_model(joint_target=joint_target)
    window = {k: v[:, :steps].clone() for k, v in inputs.items()}
    with torch.inference_mode():
        full = model({k: v.clone() for k, v in window.items()}, return_aux=True)
        single = model({k: v.clone() for k, v in window.items()}, return_aux=True, single_step=True)
        tokens, key_features = model.encode_frames({k: v.clone() for k, v in window.items()})
        fast_tokens, fast_key_features = model.encode_frames({k: v.clone() for k, v in window.items()}, inference=True)
        encoded = model.forward_encoded(tokens, key_features)
    torch.testing.assert_close(fast_tokens, tokens, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fast_key_features, key_features)
    for output in (single, encoded):
        torch.testing.assert_close(output.state_prediction, full.state_prediction[:, -1:], atol=1e-6, rtol=1e-5)
        assert output.auxiliary_predictions.keys() == full.auxiliary_predictions.keys()
        for name, value in output.auxiliary_predictions.items():
            torch.testing.assert_close(value, full.auxiliary_predictions[name][:, -1:], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("reset_history", ["replicate", "episode_start"])
def test_streaming_key_head_solver_matches_full_window_through_growth_eviction_and_row_rebuild(reset_history):
    """The streamed key head reads only its recent key frames, yet must match the full-window prediction.

    With ``episode_start``, a reset row must instead match a window that starts at its first post-reset frame.
    """
    from collections import deque

    from isaaclab_neural.solvers.transformer_neural_solver import TransformerNeuralSolver

    model, inputs, _, _ = make_key_head_model({"position_encoding": "relative_lag", "attention_spans": [3]})
    model.eval()
    for name, parameter in model.named_parameters():
        if name.endswith("lag_bias"):
            torch.nn.init.normal_(parameter)
    history = 4  # longer than the key head's two recent frames
    solver = TransformerNeuralSolver.__new__(TransformerNeuralSolver)
    solver.process_neural_model_inputs = lambda model_inputs: model_inputs
    solver.num_envs, solver.num_states_history, solver.torch_device = 2, history, "cpu"
    solver.use_feature_cache = solver.use_kv_cache = False
    solver.use_frame_cache, solver.neural_model = True, model
    solver.predicted_body_net_contact_forces_b = solver.feature_cache = solver.kv_cache = None
    solver.feature_cache_rebuild_env_ids = solver.kv_cache_rebuild_env_ids = None
    solver.frame_cache, solver.frame_cache_rebuild_env_ids, solver.stream_cache = deque(maxlen=history), None, None
    solver.states_history = deque(maxlen=history)
    solver.reset_history = reset_history

    torch.manual_seed(1)
    since_reset: list[dict] = []
    with torch.inference_mode():
        for step in range(9):
            frame = {name: value[:, step % 4].clone() for name, value in inputs.items()}
            for name in ("states_embedding", "joint_f", "gravity_dir"):
                frame[name] += torch.randn_like(frame[name])
            solver.states_history.append(frame)
            if step == 6:
                # A partial reset rewrites row 0's raw history; its cached encodings must be rebuilt.
                for entry in solver.states_history:
                    for name, value in frame.items():
                        entry[name][0] = value[0]
                solver._mark_history_rows_for_rebuild(torch.tensor([0]))
            if step >= 6:
                since_reset.append({name: value[:1].clone() for name, value in frame.items()})
            window = {name: torch.stack([entry[name] for entry in solver.states_history], dim=1) for name in inputs}
            expected = model(window, single_step=True)
            if reset_history == "episode_start" and since_reset:
                fresh = {name: torch.stack([entry[name] for entry in since_reset], dim=1) for name in inputs}
                expected[:1] = model(fresh, single_step=True)
            actual = solver._model_forward(solver.get_neural_model_inputs())
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert solver.stream_cache is not None


def test_key_head_assembles_fp32_state_under_reduced_precision_autocast():
    model, inputs, _, _ = make_key_head_model()
    reference = model({k: v.clone() for k, v in inputs.items()})
    with torch.autocast("cpu", dtype=torch.bfloat16):
        reduced = model({k: v.clone() for k, v in inputs.items()})
    assert reduced.dtype == torch.float32
    torch.testing.assert_close(reduced, reference, rtol=0.1, atol=0.1)


def test_key_head_checkpoint_roundtrip_and_cached_inference_rejection(tmp_path):
    model, inputs, input_cfg, network = make_key_head_model()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, "SO101-Keyboard-Fixed", {"inputs": input_cfg, "network": network})
    solver = SimpleNamespace(
        get_neural_model_inputs=lambda: inputs,
        prediction_dim=10,
        contact_mode="newton_native",
        num_contact_bodies_per_env=6,
    )
    restored = reconstruct_model_from_checkpoint(
        torch.load(path, map_location="cpu", weights_only=False), solver, device="cpu"
    )
    torch.testing.assert_close(
        model({k: v.clone() for k, v in inputs.items()}), restored({k: v.clone() for k, v in inputs.items()})
    )
    restored.eval()
    with torch.no_grad(), pytest.raises(RuntimeError, match="key_head"):
        restored.forward_feature_cached({k: v.clone() for k, v in inputs.items()})
    bad = {**network, "key_head": {**KEY_HEAD, "num_keys": 5}}
    with pytest.raises(ValueError, match="Key joints"):
        ModelMixedInput(inputs, 10, input_cfg, bad, device="cpu", num_bodies=6)


def test_pooled_key_velocity_weights_match_concatenated_samples_and_key_layout_is_checked():
    from isaaclab_neural.train.trainers import keyboard_pooled_key_velocity_weights, validate_keyboard_key_head_layout

    samples = torch.randn(500, 8, dtype=torch.float64) * torch.tensor([1, 2, 3, 4, 0.1, 0.2, 5, 0.3]) + torch.arange(
        8.0
    )
    rms = RunningMeanStd(shape=(8,), device="cpu")
    rms.update(samples.float(), batch_dim=True)
    weights = keyboard_pooled_key_velocity_weights(torch.ones(8), rms, num_q=4, key_q_indices=[2, 3])
    expected = 1.0 / torch.sqrt(samples[:, [6, 7]].reshape(-1).var(unbiased=False) + 1e-5)
    torch.testing.assert_close(weights[[6, 7]].double(), expected.expand(2), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(weights[:6], torch.ones(6))

    model, _, _, _ = make_key_head_model()
    metadata = {
        "asset_order": ["robot", "keyboard"],
        "num_q": 5,
        "key_q_indices": [2, 3, 4],
        "joint_names": {"keyboard": ["key_0_joint", "key_1_joint", "key_2_joint"], "robot": ["a", "b"]},
        "body_names": {"keyboard": ["base", "key_0", "key_1", "key_2"], "robot": ["base", "link"]},
    }
    validate_keyboard_key_head_layout(model.key_head, metadata)
    metadata["body_names"]["keyboard"] = ["base", "key_0", "key_2", "key_1"]
    with pytest.raises(ValueError, match="different keys"):
        validate_keyboard_key_head_layout(model.key_head, metadata)


def test_keyboard_episode_outcomes_are_exclusive_and_ordered():
    from isaaclab_neural.eval.keyboard_policy import summarize_keyboard_episodes

    def episode(**overrides):
        base = {
            "success": False,
            "excessive_contact": False,
            "abnormal_robot": False,
            "length": 10,
            "start_distance": 3,
            "final_distance": 3,
            "initial_typed_len": 0,
            "target": [4, 5, 6, -1],
            "typed": [-1, -1, -1, -1],
        }
        return {**base, **overrides}

    episodes = [
        episode(success=True, final_distance=0, typed=[4, 5, 6, -1]),
        episode(success=True, excessive_contact=True),  # a contact termination is never a clean success
        episode(abnormal_robot=True),
        episode(),  # nothing registered
        episode(final_distance=4, typed=[4, 9, -1, -1]),  # wrong key in the buffer
        episode(final_distance=1, typed=[4, 5, -1, -1]),  # correct but incomplete prefix
    ]
    assert summarize_keyboard_episodes(episodes) == {
        "episodes": 6,
        "length_sum": 60,
        "success": 1,
        "excessive_contact": 1,
        "abnormal_robot": 1,
        "timeout_no_key": 1,
        "timeout_wrong_key": 1,
        "timeout_incomplete": 1,
    }


def test_keyboard_training_eval_requires_closed_loop_policy_mode(tmp_path):
    from isaaclab_neural.train.trainers import VanillaTrainer

    trainer = object.__new__(VanillaTrainer)
    trainer.robot_force_head_enabled = True
    with pytest.raises(ValueError, match="keyboard_policy"):
        trainer._init_evaluator({"eval": {"interval": 5, "mode": "dataset"}}, {})
    with pytest.raises(FileNotFoundError):
        trainer._init_evaluator(
            {"eval": {"interval": 5, "mode": "keyboard_policy", "policy_checkpoint": "missing.pt"}}, {}
        )
    policy = tmp_path / "policy.pt"
    policy.touch()
    trainer._init_evaluator(
        {"eval": {"mode": "keyboard_policy", "policy_checkpoint": str(policy)}}, {"eval_interval": 5}
    )
    assert trainer.eval_interval == 5 and trainer.evaluator is None
    assert trainer.keyboard_policy_eval == {"checkpoint": policy, "control_steps": 450, "seed": 43}
    trainer._init_evaluator({"eval": {"interval": 5, "mode": "keyboard_policy"}}, {"eval_interval": 0})
    assert trainer.keyboard_policy_eval is None


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
    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="scale"):
            keyboard_position_loss_weights(weights, num_q=4, key_q_indices=[2, 3], scale_m=scale)
    with pytest.raises(ValueError, match="indices"):
        keyboard_position_loss_weights(weights, num_q=4, key_q_indices=[4], scale_m=0.0001)


def test_keyboard_state_group_scale_matches_group_mean_reference():
    from isaaclab_neural.train.trainers import keyboard_state_group_loss_scale

    errors = torch.rand(5, 8, dtype=torch.float64)
    robot, keys = [0, 1, 4, 5], [2, 3, 6, 7]
    for robot_weight, key_weight in ((0.5, 0.5), (1.0, 0.0), (0.2, 3.0)):
        scale = keyboard_state_group_loss_scale(
            num_q=4, key_q_indices=[2, 3], robot_weight=robot_weight, key_weight=key_weight
        ).double()
        expected = robot_weight * errors[:, robot].mean() + key_weight * errors[:, keys].mean()
        torch.testing.assert_close((errors * scale).mean(), expected)
    for weights in ((0.0, 0.0), (-1.0, 1.0), (float("nan"), 1.0), (float("inf"), 1.0)):
        with pytest.raises(ValueError, match="weights"):
            keyboard_state_group_loss_scale(
                num_q=4, key_q_indices=[2, 3], robot_weight=weights[0], key_weight=weights[1]
            )
    for indices in ([4], [2, 2], [0, 1, 2, 3]):
        with pytest.raises(ValueError, match="indices"):
            keyboard_state_group_loss_scale(num_q=4, key_q_indices=indices, robot_weight=1.0, key_weight=1.0)


def test_keyboard_state_group_loss_excludes_zero_weight_group_and_logs_group_errors():
    from isaaclab_neural.train.trainers import VanillaTrainer, keyboard_state_group_loss_scale

    torch.manual_seed(0)
    inputs = {"states_embedding": torch.randn(2, 3, 4)}
    model = ModelMixedInput(inputs, 4, {"low_dim": ["states_embedding"]}, model_config(), device="cpu", num_bodies=116)
    model.set_output_rms(RunningMeanStd(shape=(4,), device="cpu"))
    model.set_robot_net_normal_force_rms(RunningMeanStd(shape=(7, 3), device="cpu"))
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.neural_model = model
    trainer.force_head_enabled = False
    trainer.robot_force_head_enabled = True
    trainer.robot_force_loss_weight = 1.0
    trainer.loss_weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    trainer.loss_func = torch.nn.MSELoss()
    trainer.neural_env = SimpleNamespace(frame_dt=0.01)
    trainer.neural_solver = SimpleNamespace(
        dof_q_per_env=2, convert_prediction_to_next_states=lambda **kwargs: kwargs["prediction"], wrap2PI=lambda _: None
    )
    trainer.cfg = {"keyboard_dataset_contracts": {"train": {"num_q": 2, "key_q_indices": [1]}}}
    data = {
        **inputs,
        "states": torch.zeros(2, 3, 4),
        "next_states": torch.zeros(2, 3, 4),
        "prediction_target": torch.randn(2, 3, 4),
        LABEL_KEY: torch.zeros(2, 3, 7, 3),
    }
    changed = {key: value.clone() for key, value in data.items()}
    changed["prediction_target"][..., [1, 3]] += 5.0

    baseline, metrics = trainer.compute_loss(data, train=False)
    baseline_changed, _ = trainer.compute_loss(changed, train=False)
    assert not torch.isclose(baseline, baseline_changed)
    assert {f"{group}_weighted_state_MSE" for group in ("robot_q", "robot_qd", "key_q", "key_qd")} <= metrics.keys()

    trainer.state_loss_group_scale = keyboard_state_group_loss_scale(
        num_q=2, key_q_indices=[1], robot_weight=1.0, key_weight=0.0
    )
    robot_only, _ = trainer.compute_loss(data, train=False)
    robot_only_changed, _ = trainer.compute_loss(changed, train=False)
    torch.testing.assert_close(robot_only, robot_only_changed)


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


def test_keyboard_policy_runner_is_built_outside_the_training_process_group(monkeypatch, tmp_path):
    import isaaclab_neural.eval.keyboard_policy as keyboard_policy

    import isaaclab_tasks.utils.hydra as hydra
    import isaaclab_tasks.utils.parse_cfg as parse_cfg

    seen = {}
    monkeypatch.setattr(parse_cfg, "resolve_task_config", lambda *args, **kwargs: (None, object()))
    monkeypatch.setattr(hydra, "resolve_presets", lambda agent, presets: agent)
    monkeypatch.setattr(keyboard_policy, "load_saved_agent_cfg_for_policy", lambda *args, **kwargs: {})

    def build(env, runner_cfg, checkpoint, device):
        seen["world_size"] = keyboard_policy.os.environ.get("WORLD_SIZE")
        return "policy"

    monkeypatch.setattr(keyboard_policy, "build_policy_env", build)
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert keyboard_policy.load_keyboard_policy(None, tmp_path / "p.pt", device="cpu", seed=0) == "policy"
    assert seen["world_size"] == "1"
    assert keyboard_policy.os.environ["WORLD_SIZE"] == "8"
