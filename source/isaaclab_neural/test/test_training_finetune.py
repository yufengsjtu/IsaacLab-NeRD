# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint and supervised-context contracts for dynamics fine-tuning."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_neural.train.arguments import get_parser
from isaaclab_neural.train.trainers import VanillaTrainer, masked_dynamics_loss, validate_finetune_checkpoint


def test_checkpoint_mode_defaults_to_resume():
    assert get_parser().parse_args([]).checkpoint_mode == "resume"
    assert get_parser().parse_args(["--checkpoint-mode", "finetune"]).checkpoint_mode == "finetune"


@pytest.mark.parametrize("mode", ["resume", "finetune"])
def test_resume_restores_training_progress_but_finetune_does_not(mode):
    model = torch.nn.Linear(2, 1)
    old_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model(torch.ones(1, 2)).sum().backward()
    old_optimizer.step()
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.checkpoint_mode = mode
    trainer.device = "cpu"
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    trainer.start_epoch = 0
    trainer.best_valid_losses = {}
    trainer.best_eval_error = float("inf")
    trainer._checkpoint = {
        "optimizer_state_dict": old_optimizer.state_dict(),
        "epoch": 399,
        "best_valid_losses": {"old": 0.1},
        "best_eval_error": 0.2,
        "loss_weights": torch.ones(3),
    }
    trainer._restore_training_state()
    if mode == "resume":
        assert trainer.start_epoch == 400
        assert trainer.optimizer.param_groups[0]["lr"] == 0.1
        assert trainer.optimizer.state
        assert trainer.best_valid_losses == {"old": 0.1}
    else:
        assert trainer.start_epoch == 0
        assert trainer.optimizer.param_groups[0]["lr"] == 1e-5
        assert not trainer.optimizer.state
        assert trainer.best_valid_losses == {}
        assert trainer.best_eval_error == float("inf")


def test_finetune_requires_frozen_complete_statistics():
    from isaaclab_neural.utils.running_mean_std import RunningMeanStd

    model = SimpleNamespace(
        input_rms=RunningMeanStd((3,), device="cpu"),
        output_rms=RunningMeanStd((3,), device="cpu"),
        predict_body_net_contact_forces=False,
    )
    checkpoint = {"loss_weights": torch.ones(3)}
    validate_finetune_checkpoint(checkpoint, model, update_statistics=False)
    with pytest.raises(ValueError, match="statistics"):
        validate_finetune_checkpoint(checkpoint, model, update_statistics=True)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_finetune_checkpoint(None, model, update_statistics=False)
    with pytest.raises(ValueError, match="loss_weights"):
        validate_finetune_checkpoint({}, model, update_statistics=False)
    model.output_rms.var.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        validate_finetune_checkpoint(checkpoint, model, update_statistics=False)
    model.output_rms = None
    with pytest.raises(ValueError, match="RMS"):
        validate_finetune_checkpoint(checkpoint, model, update_statistics=False)


@pytest.mark.parametrize("loss", [torch.nn.MSELoss(), torch.nn.L1Loss(), torch.nn.SmoothL1Loss(beta=0.2)])
def test_masked_loss_ignores_unlabelled_targets_and_normalizes_per_window(loss):
    prediction = torch.zeros(2, 3, 2, requires_grad=True)
    target = torch.tensor([[[99.0, 99.0], [99.0, 99.0], [2.0, 2.0]], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])
    mask = torch.tensor([[False, False, True], [True, True, True]])
    expected = (loss(prediction[0, -1], target[0, -1]) + loss(prediction[1], target[1])) / 2
    actual = masked_dynamics_loss(prediction, target, loss, mask)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert not prediction.grad[0, :-1].any()
    changed = target.clone()
    changed[0, :-1] = float("nan")
    torch.testing.assert_close(masked_dynamics_loss(prediction, changed, loss, mask), actual)
    torch.testing.assert_close(masked_dynamics_loss(prediction, target, loss), loss(prediction, target))


def test_mask_rejects_empty_or_wrong_shape():
    pred = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match="label"):
        masked_dynamics_loss(pred, pred, torch.nn.MSELoss(), torch.zeros(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="shape"):
        masked_dynamics_loss(pred, pred, torch.nn.MSELoss(), torch.ones(2, dtype=torch.bool))


def test_explicit_physical_dt_overrides_control_dt_without_changing_default():
    seen = []
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.force_head_enabled = False
    trainer.device = "cpu"
    trainer.non_blocking_data_transfer = False
    trainer.neural_env = SimpleNamespace(frame_dt=0.02)

    def targets(**kwargs):
        seen.append(kwargs["dt"])
        return kwargs["next_states"]

    trainer.neural_solver = SimpleNamespace(
        process_neural_model_inputs=lambda data: None, convert_next_states_to_prediction=targets
    )
    data = {"states": torch.zeros(2, 3, 4), "next_states": torch.zeros(2, 3, 4)}
    trainer.preprocess_data_batch(dict(data))
    trainer.training_dt = 0.005
    trainer.preprocess_data_batch(dict(data))
    assert seen == [0.02, 0.005]


@pytest.mark.parametrize("clip", [False, True])
def test_microbatches_accumulate_by_anchor_count_and_step_once(monkeypatch, clip):
    from contextlib import nullcontext

    import isaaclab_neural.train.trainers as training

    monkeypatch.setattr(training, "TimeProfiler", lambda *a, **k: nullcontext())
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.device = "cpu"
    trainer.is_main_process = False
    trainer.truncate_grad = clip
    trainer.grad_norm = 2.0
    trainer.time_report = None
    trainer.neural_model = torch.nn.Linear(1, 1, bias=False)
    trainer.neural_model.weight.data.zero_()
    trainer.optimizer = torch.optim.SGD(trainer.neural_model.parameters(), lr=0.1)
    trainer.preprocess_data_batch = lambda data: data
    calls = []
    real_step = trainer.optimizer.step
    trainer.optimizer.step = lambda: (calls.append(1), real_step())

    def loss(data, train):
        predicted = trainer.neural_model(data["states"][:, -1])
        value = (predicted - data["target"]).square().mean()
        return value, {"mse": float(value.detach())}

    trainer.compute_loss = loss
    microbatches = [
        {"states": torch.ones(1, 1, 1), "target": torch.ones(1, 1)},
        {"states": torch.ones(3, 10, 1), "target": torch.full((3, 1), 3.0)},
    ]
    loader = [microbatches]
    value, itemized, grads = trainer.one_epoch(True, loader, iter(loader), 1, distributed_reduce=False)
    assert calls == [1]
    assert value == itemized["mse"] == 7.0
    assert grads["grad_norm_before_clip"] == 5.0
    torch.testing.assert_close(trainer.neural_model.weight, torch.tensor([[0.2 if clip else 0.5]]))
