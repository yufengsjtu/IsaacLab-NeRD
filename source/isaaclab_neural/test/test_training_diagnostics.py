# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for lightweight NeRD training diagnostics."""

import math
import sys
from types import SimpleNamespace

import pytest
import torch
from isaaclab_neural.train.trainers import VanillaTrainer
from isaaclab_neural.train.training_diagnostics import (
    mean_predictor_loss,
    parameter_change_metrics,
    snapshot_trainable_parameters,
    state_error_metrics,
    tensor_distribution_metrics,
)
from isaaclab_neural.utils.logger import Logger
from isaaclab_neural.utils.time_report import TimeReport


def test_state_error_metrics_split_position_and_velocity_errors():
    predicted = torch.tensor([[[1.0, 3.0, 2.0, 6.0]]])
    target = torch.zeros_like(predicted)

    metrics = state_error_metrics(predicted, target, dof_q=2)

    assert metrics["state_MSE"] == pytest.approx(12.5)
    assert metrics["q_MSE"] == pytest.approx(5.0)
    assert metrics["qd_MSE"] == pytest.approx(20.0)
    assert metrics["state_L2"] == pytest.approx(math.sqrt(50.0))
    assert metrics["q_error_norm"] == pytest.approx(math.sqrt(10.0))
    assert metrics["qd_error_norm"] == pytest.approx(math.sqrt(40.0))


def test_tensor_distribution_metrics_detect_constant_sample_representation():
    values = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )

    metrics = tensor_distribution_metrics(values, "hidden", include_positive_fraction=True)

    assert metrics["hidden_mean"] == pytest.approx(0.5)
    assert metrics["hidden_std"] == pytest.approx(0.5)
    assert metrics["hidden_sample_variance"] == pytest.approx(0.0)
    assert metrics["hidden_positive_fraction"] == pytest.approx(0.5)


def test_parameter_change_metrics_reports_epoch_displacement():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0]]))
    snapshot = snapshot_trainable_parameters(model)
    with torch.no_grad():
        model.weight[0, 0] += 1.0

    metrics = parameter_change_metrics(snapshot, model)

    assert metrics["parameter_update_norm"] == pytest.approx(1.0)
    assert metrics["parameter_norm"] == pytest.approx(math.sqrt(32.0))
    assert metrics["relative_parameter_update_norm"] == pytest.approx(1.0 / math.sqrt(32.0))


def test_mean_predictor_loss_uses_current_dataset_weights():
    target_variance = torch.tensor([4.0, 9.0])
    loss_weights = torch.tensor([0.5, 1.0 / 3.0])

    assert mean_predictor_loss(target_variance, loss_weights) == pytest.approx(1.0)


def test_time_report_exposes_numeric_seconds():
    report = TimeReport()
    report.add_timer("train_total")
    report.timers["train_total"].total_time = 1.25

    assert report.as_dict() == {"train_total": 1.25}


def test_time_report_exposes_sampled_latency_statistics_only_when_enabled():
    report = TimeReport(record_samples=True)
    report.add_timer("train_step")
    report.timers["train_step"].samples = [0.01, 0.02, 0.03, 0.04]

    assert report.sample_statistics() == {
        "train_step_count": 4.0,
        "train_step_mean_seconds": pytest.approx(0.025),
        "train_step_p50_seconds": pytest.approx(0.025),
        "train_step_p95_seconds": pytest.approx(0.0385),
    }
    assert report.sample_statistics(skip_first=2) == {
        "train_step_count": 2.0,
        "train_step_mean_seconds": pytest.approx(0.035),
        "train_step_p50_seconds": pytest.approx(0.035),
        "train_step_p95_seconds": pytest.approx(0.0395),
    }
    report.reset_timer()
    assert report.sample_statistics() == {}

    disabled = TimeReport(record_samples=False)
    disabled.add_timer("train_step")
    disabled.timers["train_step"].samples = [1.0]
    assert disabled.sample_statistics() == {}


def test_time_report_cuda_events_defer_synchronization_until_finalize(monkeypatch):
    timestamps_ms = iter((10.0, 35.0))
    synchronize_calls = []

    class FakeEvent:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.timestamp_ms = None

        def record(self):
            self.timestamp_ms = next(timestamps_ms)

        def elapsed_time(self, other):
            return other.timestamp_ms - self.timestamp_ms

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronize_calls.append(True))
    report = TimeReport(cuda_event_timing=True, record_samples=True)
    report.add_timer("train_step")

    report.start_timer("train_step")
    report.end_timer("train_step")

    assert synchronize_calls == []
    report.finalize()
    assert synchronize_calls == [True]
    assert report.timers["train_step"].samples == pytest.approx([0.025])


def test_time_report_rejects_boundary_sync_with_cuda_events():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TimeReport(cuda_synchronize=True, cuda_event_timing=True)


def test_time_report_can_sample_only_selected_steps(monkeypatch):
    timestamps = iter((1.0, 1.25))
    monkeypatch.setattr("isaaclab_neural.utils.time_report.time.perf_counter", lambda: next(timestamps))
    report = TimeReport(record_samples=True)
    report.add_timer("train_forward_loss")

    report.set_timer_enabled("train_forward_loss", False)
    report.start_timer("train_forward_loss")
    report.end_timer("train_forward_loss")
    report.set_timer_enabled("train_forward_loss", True)
    report.start_timer("train_forward_loss")
    report.end_timer("train_forward_loss")

    assert report.timers["train_forward_loss"].samples == pytest.approx([0.25])
    assert report.as_dict()["train_forward_loss"] == pytest.approx(0.25)


def test_logger_sets_requested_wandb_system_sampling_interval(monkeypatch):
    captured = {}
    fake_wandb = SimpleNamespace(
        Settings=lambda **kwargs: captured.setdefault("settings", kwargs),
        init=lambda **kwargs: captured.setdefault("init", kwargs),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    logger = Logger()

    logger.init_wandb(
        "project",
        "run",
        system_stats_interval_seconds=1.0,
    )

    assert captured["settings"] == {
        "console": "wrap",
        "x_stats_sampling_interval": 1.0,
    }


class _DiagnosticHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_net = torch.nn.ReLU()


class _DiagnosticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DiagnosticHead()
        self.contact_set_encoder = torch.nn.Identity()


class _LoopModel(torch.nn.Module):
    is_rnn = False

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, data):
        return self.linear(data["feature"])


class _LoopSolver:
    dof_q_per_env = 1

    def process_neural_model_inputs(self, _data):
        pass

    def convert_next_states_to_prediction(self, *, states, next_states, dt):
        del states, dt
        return next_states

    def convert_prediction_to_next_states(self, *, states, prediction, dt):
        del states, dt
        return prediction

    def wrap2PI(self, _states):
        pass


def test_trainer_hooks_capture_forward_distributions_without_changing_outputs():
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.neural_model = _DiagnosticModel()
    trainer._capture_forward_diagnostics = False
    trainer._forward_diagnostic_outputs = {}
    trainer._diagnostic_handles = []
    trainer._register_diagnostic_hooks()
    trainer._set_diagnostic_capture(True)

    contact_input = torch.tensor([[[-1.0, 2.0]]])
    hidden_input = torch.tensor([[[-2.0, 3.0]]])
    contact_output = trainer.neural_model.contact_set_encoder(contact_input)
    hidden_output = trainer.neural_model.model.feature_net(hidden_input)
    metrics = trainer._collect_forward_diagnostics(torch.tensor([[[1.0, 1.0]]]))

    assert torch.equal(contact_output, contact_input)
    assert torch.equal(hidden_output, torch.tensor([[[0.0, 3.0]]]))
    assert metrics["prediction_sample_variance"] == pytest.approx(0.0)
    assert metrics["final_hidden_positive_fraction"] == pytest.approx(0.5)
    assert metrics["contact_representation_sample_variance"] == pytest.approx(0.0)
    for handle in trainer._diagnostic_handles:
        handle.remove()


def test_one_epoch_reports_physical_metrics_clip_fraction_and_phase_timers():
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.neural_model = _LoopModel()
    trainer.neural_solver = _LoopSolver()
    trainer.neural_env = SimpleNamespace(frame_dt=1.0)
    trainer.optimizer = torch.optim.SGD(trainer.neural_model.parameters(), lr=0.1)
    trainer.loss_func = torch.nn.MSELoss()
    trainer.loss_weights = 1.0
    trainer.batch_size = 1
    trainer.device = "cpu"
    trainer.non_blocking_data_transfer = False
    trainer.diagnostics_enabled = False
    trainer.diagnostic_batches_per_epoch = 1
    trainer._capture_forward_diagnostics = False
    trainer._forward_diagnostic_outputs = {}
    trainer.truncate_grad = True
    trainer.grad_norm = 0.0
    trainer.is_main_process = True
    trainer.is_distributed = False
    trainer.train_dataset = None
    trainer.time_report = TimeReport(record_samples=True)
    trainer.time_report.add_timers(
        [
            "train_step",
            "train_dataloader",
            "train_data_transfer",
            "train_preprocess",
            "train_forward_loss",
            "train_metrics",
            "train_backward_ddp",
            "train_gradient_diagnostics",
            "train_zero_grad",
            "train_optimizer_step",
            "train_bookkeeping",
        ]
    )
    batch = {
        "feature": torch.ones((1, 1, 2)),
        "states": torch.zeros((1, 1, 2)),
        "next_states": torch.tensor([[[1.0, 2.0]]]),
        "contact_tokens": torch.zeros((1, 1, 1, 17)),
    }
    dataloader = [batch]

    loss, metrics, grad_info, diagnostics = trainer.one_epoch(
        train=True,
        dataloader=dataloader,
        dataloader_iter=iter(dataloader),
        num_batches=3,
    )

    assert loss == pytest.approx(2.5)
    assert metrics["state_MSE"] == pytest.approx(2.5)
    assert metrics["q_MSE"] == pytest.approx(1.0)
    assert metrics["qd_MSE"] == pytest.approx(4.0)
    assert grad_info["clip_fraction"] == pytest.approx(1.0)
    assert diagnostics == {}
    assert trainer.time_report.as_dict()["train_forward_loss"] > 0.0
    assert trainer.time_report.sample_statistics()["train_step_count"] == 3.0
    assert trainer.train_iterator_reset_steps == [1, 2]


def test_trainer_captures_peak_gpu_memory_without_allocating(monkeypatch):
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.device = "cuda:0"
    trainer.is_distributed = False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 3 * 1024**3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 4 * 1024**3)

    trainer._capture_peak_gpu_memory()

    assert trainer.peak_gpu_memory_gib_local == pytest.approx(3.0)
    assert trainer.peak_gpu_memory_gib_mean == pytest.approx(3.0)
    assert trainer.peak_gpu_memory_gib_max == pytest.approx(3.0)
    assert trainer.peak_gpu_reserved_gib_local == pytest.approx(4.0)
    assert trainer.peak_gpu_reserved_gib_mean == pytest.approx(4.0)
    assert trainer.peak_gpu_reserved_gib_max == pytest.approx(4.0)


def test_trainer_gathers_rank_samples_and_uses_stepwise_slowest_rank(monkeypatch):
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.device = "cpu"
    trainer.is_distributed = True
    trainer.world_size = 2

    def fake_all_gather(gathered, _local):
        gathered[0].copy_(torch.tensor([[0.10, 0.30], [0.04, 0.20]], dtype=torch.float64))
        gathered[1].copy_(torch.tensor([[0.20, 0.25], [0.08, 0.10]], dtype=torch.float64))

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    by_rank, critical_path = trainer._gather_profile_samples(
        {
            "train_step": [0.10, 0.30],
            "train_forward_loss": [0.04, 0.20],
        }
    )

    torch.testing.assert_close(
        torch.tensor(by_rank["train_step"]),
        torch.tensor([[0.10, 0.30], [0.20, 0.25]]),
    )
    assert critical_path["train_step"] == pytest.approx([0.20, 0.30])
    assert critical_path["train_forward_loss"] == pytest.approx([0.08, 0.20])


class _CaptureLogger:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, _step):
        self.values[name] = value


def test_finalize_profile_separates_full_steps_from_sampled_phases():
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.device = "cpu"
    trainer.is_distributed = False
    trainer.profile_record_samples = True
    trainer.profile_warmup_steps = 1
    trainer.profile_phase_steps = 2
    trainer.profile_phase_timer_names = ("train_forward_loss",)
    trainer.train_iterator_reset_steps = []
    trainer.profile_measurement_start_unix = 1.0
    trainer.time_report = TimeReport(record_samples=True)
    trainer.time_report.add_timers(["train_total", "train_measurement_window", "train_step", "train_forward_loss"])
    trainer.time_report.timers["train_total"].total_time = 1.0
    trainer.time_report.timers["train_measurement_window"].total_time = 0.8
    trainer.time_report.timers["train_step"].samples = [0.1, 0.2, 0.3, 0.4]
    trainer.time_report.timers["train_forward_loss"].samples = [0.05, 0.06]

    trainer._finalize_train_profile()

    assert trainer.measurement_window_seconds_by_rank == pytest.approx([0.8])
    assert trainer.measurement_window_seconds_max == pytest.approx(0.8)
    assert trainer.profile_critical_path_samples["train_step"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert trainer.profile_phase_train_step_samples == pytest.approx([0.2, 0.3])
    assert trainer.profile_critical_path_samples["train_forward_loss"] == pytest.approx([0.05, 0.06])
    assert trainer.profile_phase_sample_steps == pytest.approx(2.0)


def test_profile_throughput_uses_continuous_measurement_window():
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.logger = _CaptureLogger()
    trainer.train_total_seconds_local = 1.0
    trainer.train_total_seconds_mean = 1.0
    trainer.train_total_seconds_max = 1.0
    trainer.measurement_window_seconds_local = 0.5
    trainer.measurement_window_seconds_mean = 0.6
    trainer.measurement_window_seconds_max = 0.8
    trainer.measurement_window_seconds_by_rank = [0.5]
    trainer.peak_gpu_memory_gib_local = 0.0
    trainer.peak_gpu_memory_gib_mean = 0.0
    trainer.peak_gpu_memory_gib_max = 0.0
    trainer.peak_gpu_reserved_gib_local = 0.0
    trainer.peak_gpu_reserved_gib_mean = 0.0
    trainer.peak_gpu_reserved_gib_max = 0.0
    trainer.train_iterator_reset_count_local = 0.0
    trainer.train_iterator_reset_count_max = 0.0
    trainer.profile_measurement_start_unix = 1.0
    trainer.profile_measurement_end_unix = 2.0
    trainer.profile_warmup_steps = 1
    trainer.profile_phase_steps = 1
    trainer.profile_phase_train_step_samples = [0.2]
    trainer.profile_critical_path_samples = {
        "train_step": [0.1, 0.2, 0.3],
        "train_forward_loss": [0.04],
    }
    trainer.profile_samples_by_rank = {"train_step": [[0.1, 0.2, 0.3]]}
    trainer.train_iterator_reset_steps = []
    trainer.num_train_batches = 3
    trainer.global_batch_windows = 8
    trainer.sample_sequence_length = 10

    trainer._log_profile_metrics(epoch=0)

    assert trainer.logger.values["performance/steady_optimizer_steps_per_second/epoch"] == pytest.approx(2.5)
    assert trainer.logger.values["performance/steady_windows_per_second/epoch"] == pytest.approx(20.0)
    assert trainer.logger.values["performance/steady_train_step_ms_mean/epoch"] == pytest.approx(300.0)
    assert trainer.logger.values["performance/phase_sample_train_step_ms_mean/epoch"] == pytest.approx(200.0)
    assert trainer.logger.values["timing/critical_path_train_forward_loss_ms_mean/epoch"] == pytest.approx(40.0)
    assert trainer.logger.values["timing/critical_path_train_forward_loss_fraction/epoch"] == pytest.approx(0.2)
