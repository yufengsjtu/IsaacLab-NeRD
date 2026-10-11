# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Epoch control preserves logical gradient weights, budgets and matching optimizer snapshots."""

import copy

import pytest
import torch
from isaaclab_neural.data.opd import CurrentEpochSampler
from isaaclab_neural.train.opd_sequence import OpdEpochConfig, fit_opd_epochs, sequence_pd_effort


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Warp/Torch stream interop.")
@pytest.mark.parametrize("use_default_stream", [True, False])
def test_native_forward_uses_the_actual_warp_stream(monkeypatch, use_default_stream):
    from types import SimpleNamespace

    import warp as wp
    from isaaclab_neural.solvers.neural_solver import NeuralSolver
    from isaaclab_neural.utils import newton_utils

    device = wp.get_device("cuda:0")
    stream = wp.stream_from_torch(torch.cuda.default_stream("cuda:0")) if use_default_stream else wp.Stream(device)
    seen = []
    data = {"states": torch.ones(1, 1, 2, device="cuda:0"), "root_body_q": torch.zeros(1, 1, 7, device="cuda:0")}

    def predict(inputs):
        seen.append(torch.cuda.current_stream("cuda:0").cuda_stream)
        return torch.zeros_like(inputs["states"])

    solver = SimpleNamespace(
        device=device,
        torch_device="cuda:0",
        neural_model=object(),
        model=object(),
        _update_states=lambda *args: None,
        _before_model_forward=lambda: None,
        get_neural_model_inputs=lambda: data,
        _model_forward=predict,
        convert_prediction_to_next_states=lambda states, prediction, dt: states,
        convert_states_back_to_world=lambda root, states: states,
        wrap2PI=lambda states: None,
        _assign_states_from_torch=lambda *args: None,
    )
    monkeypatch.setattr(newton_utils, "eval_fk", lambda *args: None)
    with wp.ScopedStream(stream):
        NeuralSolver.step(solver, object(), object(), SimpleNamespace(joint_f=None))
    assert seen == [stream.cuda_stream]


def test_rollout_profile_cannot_consume_benchmark_or_validation_sequences():
    from isaaclab_neural.train.opd_sequence import _profile_training_records

    row = {"oracle_states": torch.zeros(16, 87)}
    shard = {"metadata": {"split": "train", "source_kind": "mjwarp"}, "records": [row]}
    assert _profile_training_records(shard, 1) == [row]
    for change in ({"split": "validation"}, {"evaluation_only": True}, {"source_kind": "nerd"}):
        with pytest.raises(ValueError, match="evaluation data or synthetic"):
            _profile_training_records({**shard, "metadata": {**shard["metadata"], **change}}, 1)
    with pytest.raises(ValueError, match="complete"):
        _profile_training_records({**shard, "records": [{"oracle_states": torch.zeros(12, 87)}]}, 1)


def _fit(model, *, microbatch=1, scores=(10.0, 9.0, 8.0), epochs=2, max_epochs=2, max_steps=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.arange(14, dtype=torch.float32).reshape(7, 2) / 10
    sampler = CurrentEpochSampler(7, reference_count=7, replay_stage_sizes=[], seed=17, current_batch_size=2)
    it = iter(scores)

    def loss(source, indices):
        x = inputs[indices]
        return (model(x).flatten() - x.sum(-1)).square()

    report = fit_opd_epochs(
        model,
        optimizer,
        sampler,
        loss,
        lambda: {"loss": next(it), "admissible": True},
        OpdEpochConfig(
            min_epochs=1,
            num_epochs=epochs,
            max_epochs=max_epochs,
            max_optimizer_steps=max_steps,
            micro_batch_size=microbatch,
        ),
    )
    return optimizer, report


def test_microbatch_accumulation_matches_large_logical_batch():
    model = torch.nn.Linear(2, 1)
    other = copy.deepcopy(model)
    state = torch.get_rng_state()
    _, a = _fit(model, microbatch=1)
    assert torch.equal(state, torch.get_rng_state())
    _, b = _fit(other, microbatch=64)
    for x, y in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-7)
    assert a["optimizer_steps_executed"] == b["optimizer_steps_executed"] == 6
    assert a["coverage_per_epoch"] == [7, 7]


def test_early_stop_restores_best_model_and_matching_adam_state():
    model = torch.nn.Linear(2, 1)
    other = copy.deepcopy(model)
    optimizer, report = _fit(model, scores=(10.0, 9.0, 8.0, 8.1, 8.2), epochs=4, max_epochs=8)
    expected, _ = _fit(other)
    assert report["selected_epoch"] == 2
    assert report["epochs_completed"] == 4
    assert report["optimizer_steps_executed"] == 12
    assert report["stop_reason"] == "validation_patience"
    for x, y in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(x, y)
    for a, b in zip(optimizer.state.values(), expected.state.values()):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(a[key], b[key])


def test_step_budget_does_not_cut_a_current_epoch():
    model = torch.nn.Linear(2, 1)
    _, report = _fit(model, scores=(10.0, 9.0), epochs=4, max_epochs=8, max_steps=5)
    assert report["optimizer_steps_executed"] == 3
    assert report["coverage_per_epoch"] == [7]
    assert report["stop_reason"] == "optimizer_budget"


@pytest.mark.parametrize("stop_on_validation_gate", [True, False])
def test_nonfinite_step_rolls_back_parameters_and_optimizer(stop_on_validation_gate):
    model = torch.nn.Linear(2, 1)
    original = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sampler = CurrentEpochSampler(4, reference_count=4, replay_stage_sizes=[], seed=1, current_batch_size=2)
    with pytest.raises(FloatingPointError):
        fit_opd_epochs(
            model,
            optimizer,
            sampler,
            lambda source, indices: model(torch.ones(len(indices), 2)).flatten() * float("nan"),
            lambda: {"loss": 1.0, "admissible": True},
            OpdEpochConfig(micro_batch_size=1, stop_on_validation_gate=stop_on_validation_gate),
        )
    for key, value in original.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert not optimizer.state


@pytest.mark.parametrize("stop,epochs,selected", [(True, 1, 0), (False, 2, 2)])
def test_fixed_budget_can_reject_quality_without_stopping_or_selecting_rejected_epoch(stop, epochs, selected):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sampler = CurrentEpochSampler(4, reference_count=4, replay_stage_sizes=[], seed=1, current_batch_size=2)
    scores = iter(
        ({"loss": 3.0, "admissible": True}, {"loss": 1.0, "admissible": False}, {"loss": 2.0, "admissible": True})
    )
    report = fit_opd_epochs(
        model,
        optimizer,
        sampler,
        lambda source, indices: model(torch.ones(len(indices), 2)).flatten().square(),
        lambda: next(scores),
        OpdEpochConfig(min_epochs=1, num_epochs=2, max_epochs=2, patience_epochs=3, stop_on_validation_gate=stop),
    )
    assert report["epochs_completed"] == epochs
    assert report["selected_epoch"] == selected
    assert report["optimizer_steps_executed"] == epochs * len(sampler)
    if selected == 0:
        assert not optimizer.state
    else:
        assert all(int(state["step"]) == epochs * len(sampler) for state in optimizer.state.values())


def test_no_validation_gain_keeps_epoch_zero():
    model = torch.nn.Linear(2, 1)
    initial = copy.deepcopy(model.state_dict())
    optimizer, report = _fit(model, scores=(1.0, 1.1, 1.2))
    assert not report["accepted"] and report["selected_epoch"] == 0
    assert report["optimizer_steps_executed"] == 6
    assert not optimizer.state
    for key, value in initial.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_minimum_epochs_must_fit_budget_and_optimizer_cannot_own_policy():
    model, policy = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    sampler = CurrentEpochSampler(7, reference_count=7, replay_stage_sizes=[], seed=1, current_batch_size=2)
    with pytest.raises(ValueError, match="minimum"):
        OpdEpochConfig(min_epochs=2, max_optimizer_steps=5).validate(len(sampler))
    optimizer = torch.optim.Adam([*model.parameters(), *policy.parameters()], lr=0.001)
    with pytest.raises(ValueError, match="policy parameters"):
        fit_opd_epochs(model, optimizer, sampler, None, None, OpdEpochConfig())


@pytest.mark.parametrize("stop_on_validation_gate", [True, False])
def test_frozen_statistics_mutation_is_rejected_even_without_validation_gain(stop_on_validation_gate):
    model = torch.nn.Linear(2, 1)
    model.input_rms = torch.nn.Module()
    model.input_rms.register_buffer("mean", torch.zeros(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sampler = CurrentEpochSampler(4, reference_count=4, replay_stage_sizes=[], seed=1, current_batch_size=2)
    scores = iter((1.0, 2.0, 3.0))

    def loss(source, indices):
        model.input_rms.mean.add_(1)
        return model(torch.ones(len(indices), 2)).flatten().square()

    with pytest.raises(RuntimeError, match="statistics"):
        fit_opd_epochs(
            model,
            optimizer,
            sampler,
            loss,
            lambda: {"loss": next(scores), "admissible": True},
            OpdEpochConfig(stop_on_validation_gate=stop_on_validation_gate),
        )
    assert model.input_rms.mean.item() == 0
    assert not optimizer.state


def test_angle_wrap_preserves_runtime_values_and_gradients():
    import math

    import numpy as np
    from isaaclab_neural.train.opd_sequence import sequence_wrap_angles

    values = torch.tensor([[0.37, 7.1, -8.3]], requires_grad=True)
    mask = np.array([False, True, True])
    result = sequence_wrap_angles(values, mask)
    expected = values.detach().clone()
    expected[:, mask] -= torch.floor((expected[:, mask] + math.pi) / (2 * math.pi)) * (2 * math.pi)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_sequence_loss_terms_preserve_weighted_objective_and_gradient():
    from types import SimpleNamespace

    from isaaclab_neural.train.opd_sequence import G1SequenceStudent

    student = G1SequenceStudent.__new__(G1SequenceStudent)
    student.device = "cpu"
    student.metadata = {"physics_dt": 0.005}
    student.loss_weights = torch.ones(87)
    student.loss_weights[44] = 3
    student.solver = SimpleNamespace(
        _convert_states_w2b=lambda root, states, translation_only: states,
        convert_next_states_to_prediction=lambda states, next_states, dt: next_states - states,
    )
    initial = torch.zeros(1, 1, 87)
    initial[..., 6] = 1
    truth = initial.expand(1, 2, 87).clone()
    parameter = torch.tensor(0.2, requires_grad=True)
    direction = torch.zeros(1, 1, 87)
    direction[..., 0], direction[..., 44] = 1, 2
    prediction = truth + parameter * torch.tensor([1.0, 2.0])[None, :, None] * direction
    first = parameter * direction[:, 0]
    student.rollout = lambda records, max_physics_steps: (
        prediction,
        first,
        {"root_body_q": initial[..., :7], "states": initial},
        [],
    )
    records = [{"oracle_states": truth[0]}]
    terms = student.sequence_loss_terms(records)
    expected_local = 37 * parameter.square() / 87
    expected_sequence = 2.5 * expected_local
    torch.testing.assert_close(terms["local"], expected_local.reshape(1))
    torch.testing.assert_close(terms["sequence"], expected_sequence.reshape(1))
    loss = student.sequence_loss(records, sequence_weight=0.3)
    torch.testing.assert_close(loss, (expected_local + 0.3 * expected_sequence).reshape(1))
    gradient = torch.autograd.grad(loss.sum(), parameter)[0]
    torch.testing.assert_close(gradient, 2 * 37 * parameter / 87 * (1 + 0.3 * 2.5))


def test_root_rollout_correction_balances_physical_groups_and_detaches_truth():
    from isaaclab_neural.train.opd_sequence import sequence_root_correction_loss

    truth = torch.zeros(1, 2, 87)
    truth[..., 6] = 1
    prediction = truth.clone()
    prediction[0, :, 2] = torch.tensor([0.01, 0.02])
    prediction[..., 44] = 1
    prediction.requires_grad_()
    truth.requires_grad_()
    loss = sequence_root_correction_loss(
        prediction, truth, height_scale_m=0.01, velocity_scale_mps=1.0, upright_scale=0.1
    )
    torch.testing.assert_close(loss, torch.tensor([17.0 / 18.0]))
    gradient, target_gradient = torch.autograd.grad(loss.sum(), (prediction, truth), allow_unused=True)
    assert target_gradient is None
    torch.testing.assert_close(gradient[0, :, 2], torch.tensor([100.0 / 3.0, 200.0 / 3.0]))
    torch.testing.assert_close(gradient[0, :, 44], torch.full((2,), 1.0 / 9.0))
    mirrored = prediction.detach().clone()
    mirrored[..., 2] *= -1
    torch.testing.assert_close(
        sequence_root_correction_loss(mirrored, truth, height_scale_m=0.01, velocity_scale_mps=1.0, upright_scale=0.1),
        loss,
    )
    with pytest.raises(ValueError):
        sequence_root_correction_loss(prediction, truth, height_scale_m=0.0, velocity_scale_mps=1.0, upright_scale=0.1)


def test_training_fingerprint_includes_adam_moments_and_survives_restore():
    from isaaclab_neural.train.opd_sequence import _training_fingerprint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    expected = _training_fingerprint(model, optimizer)
    next(iter(optimizer.state.values()))["exp_avg"].add_(1)
    changed = _training_fingerprint(model, optimizer)
    assert changed["model_sha256"] == expected["model_sha256"]
    assert changed["optimizer_sha256"] != expected["optimizer_sha256"]
    optimizer.load_state_dict(saved)
    assert _training_fingerprint(model, optimizer) == expected


@pytest.mark.parametrize("pd_mode", ["torch", "newton_fma"])
def test_pd_state_path_is_differentiable_and_saturation_is_explicit(pd_mode):
    state = torch.zeros(1, 17, requires_grad=True)
    effort = sequence_pd_effort(
        state,
        torch.tensor([[0.1, 2.0]]),
        torch.tensor([10.0, 10.0]),
        torch.tensor([2.0, 2.0]),
        torch.tensor([100.0, 1.0]),
        num_q=9,
        pd_mode=pd_mode,
    )
    torch.testing.assert_close(effort[:, 6:], torch.tensor([[1.0, 1.0]]))
    effort.sum().backward()
    assert state.grad[0, 7] == -10
    assert state.grad[0, 15] == -2
    assert state.grad[0, 8] == state.grad[0, 16] == 0


def test_runtime_pd_rounding_matches_fused_reference_without_changing_legacy_default():
    state = torch.linspace(-0.7, 1.1, 85).reshape(5, 17)
    targets = torch.linspace(0.13, 0.87, 10).reshape(5, 2)
    kp, kd, limits = torch.tensor([25.6, 10.12345]), torch.tensor([0.73, 1.91]), torch.tensor([100.0, 100.0])
    position = targets - state[:, 7:9]
    damping = kd * (-state[:, 15:])
    expected = (kp.double() * position.double() + damping.double()).float().clamp(-limits, limits)
    runtime = sequence_pd_effort(state, targets, kp, kd, limits, num_q=9, pd_mode="newton_fma")
    torch.testing.assert_close(runtime[:, 6:], expected, rtol=0, atol=0)
    legacy = (kp * position - kd * state[:, 15:]).clamp(-limits, limits)
    torch.testing.assert_close(
        sequence_pd_effort(state, targets, kp, kd, limits, num_q=9)[:, 6:], legacy, rtol=0, atol=0
    )
    with pytest.raises(ValueError):
        sequence_pd_effort(state, targets, kp, kd, limits, num_q=9, pd_mode="unrecognized")


def test_runtime_rollout_preserves_recorded_anchor_effort_and_legacy_reconstruction():
    from isaaclab_neural.train.opd_sequence import G1SequenceStudent

    states = torch.zeros(1, 10, 87)
    history = {"states": states, "joint_f": torch.cat((torch.zeros(1, 10, 6), torch.full((1, 10, 37), 0.10001)), -1)}
    seen = []
    student = G1SequenceStudent.__new__(G1SequenceStudent)
    student.num_envs, student.device, student.metadata = 1, "cpu", {"decimation": 4}
    student.kp, student.kd, student.limits = torch.ones(37), torch.ones(37), torch.full((37,), 100.0)
    student._history = lambda rows: {key: value.clone() for key, value in history.items()}

    def forward(inputs):
        seen.append(inputs["joint_f"][:, -1].clone())
        return inputs["states"][:, -1], torch.zeros(1, 87), {}

    student._forward = forward
    records = [{"actions": torch.zeros(1, 37), "action_targets": torch.full((1, 37), 0.1)}]
    student.pd_mode = "newton_fma"
    student.rollout(records, max_physics_steps=1)
    torch.testing.assert_close(seen[-1], history["joint_f"][:, -1], rtol=0, atol=0)
    student.pd_mode = "torch"
    student.rollout(records, max_physics_steps=1)
    expected = sequence_pd_effort(
        states[:, -1], records[0]["action_targets"], student.kp, student.kd, student.limits, num_q=44
    )
    torch.testing.assert_close(seen[-1], expected, rtol=0, atol=0)
    assert not torch.equal(seen[-1], history["joint_f"][:, -1])
