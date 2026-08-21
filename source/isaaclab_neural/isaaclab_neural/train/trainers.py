# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trainer implementations for NeRD neural dynamics models."""

from __future__ import annotations

import inspect
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from isaaclab_neural.contacts.contact_set_schema import (
    ACTIVE15_CATEGORICAL_CHANNELS,
    CONTACT_FILTER_SOLVER_ACTIVE,
    CONTACT_REPRESENTATION_ACTIVE15,
    CONTACT_REPRESENTATION_RAW15,
    CONTACT_REPRESENTATION_TOKENS,
)
from isaaclab_neural.contacts.tensor_utils import (
    MIN_CONTACT_RMS_SAMPLES,
    ContactTokenMoments,
    MaskedContactMoments,
)
from isaaclab_neural.data import (
    collate_fn_BatchTransitionDataset,
    create_batch_transition_dataset,
    create_trajectory_dataset,
)
from isaaclab_neural.eval.training_evaluator import TrainingRolloutEvaluator
from isaaclab_neural.models.models import ModelMixedInput
from isaaclab_neural.train.training_diagnostics import (
    mean_predictor_loss,
    parameter_change_metrics,
    snapshot_trainable_parameters,
    state_error_metrics,
    tensor_distribution_metrics,
)
from isaaclab_neural.utils.checkpoint import reconstruct_model_from_checkpoint, save_checkpoint
from isaaclab_neural.utils.logger import Logger
from isaaclab_neural.utils.python_utils import format_dict, print_info, print_ok, print_warning, set_random_seed
from isaaclab_neural.utils.running_mean_std import RunningMeanStd
from isaaclab_neural.utils.time_report import TimeProfiler, TimeReport
from isaaclab_neural.utils.torch_utils import grad_norm, num_params_torch_model

NON_MODEL_DATA_KEYS = {
    "contact_token_body_ids",
    "contact_token_world_ids",
    "root_body_qd",
    "source_env_id",
    "state_world_id",
    "root_world_id",
    "contact_world_id",
    "terrain_level",
    "terrain_type",
    "env_origin",
}


def _resolve_auto_bool(value: bool | str | None, *, automatic: bool) -> bool:
    """Resolve a boolean config value that may use the string ``auto``."""
    if value is None or value == "auto":
        return automatic
    if isinstance(value, bool):
        return value
    raise ValueError(f"Expected a boolean or 'auto', got {value!r}.")


def _data_worker_init(_worker_id: int) -> None:
    """Limit intra-op parallelism inside DataLoader worker processes."""
    torch.set_num_threads(1)


def _reset_validation_iterators(valid_loaders: dict[str, Any]) -> dict[str, Any]:
    """Create fresh validation iterators so each epoch starts a new sampling order."""
    return {name: iter(loader) for name, loader in valid_loaders.items()}


@torch.no_grad()
def _synchronize_running_mean_std(rms: RunningMeanStd, epsilon: float = 1.0e-4) -> None:
    """Merge running moments across an initialized distributed process group."""
    if not dist.is_available() or not dist.is_initialized():
        return

    count_with_prior = rms.count.to(torch.float64)
    sample_count = (count_with_prior - epsilon).clamp_min(0.0)
    value_sum = rms.mean.to(torch.float64) * count_with_prior
    square_sum = (rms.var.to(torch.float64) + rms.mean.to(torch.float64).square()) * count_with_prior
    square_sum -= epsilon

    dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(value_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)

    global_count = sample_count + epsilon
    global_mean = value_sum / global_count
    global_var = (square_sum + epsilon) / global_count - global_mean.square()
    rms.mean.copy_(global_mean.to(rms.mean.dtype))
    rms.var.copy_(global_var.clamp_min(0.0).to(rms.var.dtype))
    rms.count.copy_(global_count.to(rms.count.dtype))


class TrainingEnvAdapter:
    """Expose the small environment surface needed by NeRD trainers."""

    def __init__(self, env: Any):
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        neural_adapter = getattr(self.unwrapped, "neural_adapter", None)
        if neural_adapter is None:
            raise AttributeError("Training env must expose a neural_adapter created by NeuralEnvAdapter.")
        self.neural_adapter = neural_adapter

    @property
    def solver_neural(self):
        """Active neural solver."""
        return self.neural_adapter.solver

    @property
    def frame_dt(self) -> float:
        """Environment frame step [s]."""
        return float(getattr(self.unwrapped, "step_dt", getattr(self.unwrapped, "physics_dt", 0.0)))

    @property
    def num_envs(self) -> int:
        """Number of vectorized environments."""
        return int(getattr(self.unwrapped, "num_envs", self.solver_neural.num_envs))

    @property
    def robot_name(self) -> str:
        """Robot or task name stored in training checkpoints."""
        return getattr(self.unwrapped, "robot_name", type(self.unwrapped).__name__)


def adapt_training_env(env: Any) -> Any:
    """Return an object exposing the trainer's expected env attributes."""
    if hasattr(env, "solver_neural") and hasattr(env, "frame_dt"):
        return env
    return TrainingEnvAdapter(env)


class VanillaTrainer:
    """Single-step NeRD trainer using transition datasets."""

    def __init__(self, neural_env: Any, cfg: dict[str, Any], checkpoint: dict[str, Any] | None = None, device="cuda:0"):
        algo_cfg = cfg["algorithm"]
        cli_cfg = cfg["cli"]

        self.cfg = cfg
        self.seed = algo_cfg.get("seed", 0)
        self.device = device
        self.is_distributed = bool(cli_cfg.get("distributed", False))
        self.rank = int(cli_cfg.get("rank", 0))
        self.local_rank = int(cli_cfg.get("local_rank", 0))
        self.world_size = int(cli_cfg.get("world_size", 1))
        self.is_main_process = self.rank == 0
        set_random_seed(self.seed)
        self.rng = np.random.default_rng(seed=self.seed)

        self.neural_env = adapt_training_env(neural_env)
        self.neural_solver = self.neural_env.solver_neural
        self.neural_env.neural_adapter.sync(update_history=False)

        if (
            cfg["env"]["neural_solver_cfg"].get("states_frame") == "body"
            and "gravity_dir" not in cfg["inputs"]["low_dim"]
        ):
            cfg["inputs"]["low_dim"].append("gravity_dir")
            print_warning("gravity_dir not included in low_dim inputs, added it automatically.")

        self._checkpoint = checkpoint
        if self._checkpoint is None:
            input_sample = self.neural_solver.get_neural_model_inputs()
            self.neural_model = ModelMixedInput(
                input_sample=input_sample,
                output_dim=self.neural_solver.prediction_dim,
                input_cfg=cfg["inputs"],
                network_cfg=cfg["network"],
                contact_mode=self.neural_solver.contact_mode,
                contact_representation=self.neural_solver.contact_representation,
                num_bodies=self.neural_solver.num_contact_bodies_per_env,
                device=self.device,
            )
        else:
            self.neural_model = reconstruct_model_from_checkpoint(
                self._checkpoint, self.neural_solver, device=self.device
            )

        if self.is_main_process:
            print("Model = \n", self.neural_model)
            print("# Model Parameters = ", num_params_torch_model(self.neural_model))
        self.neural_solver.set_neural_solver_model(self.neural_model)

        self.configured_batch_size = int(algo_cfg["batch_size"])
        self.batch_size = self.configured_batch_size
        self.num_valid_batches = int(algo_cfg.get("num_valid_batches", 50))
        dataset_cfg = algo_cfg["dataset"]
        self.dataset_contact_representation = dataset_cfg.get(
            "contact_representation", self.neural_solver.contact_representation
        )
        self.require_solver_active = self.neural_solver.contact_filter == CONTACT_FILTER_SOLVER_ACTIVE and (
            self.dataset_contact_representation == CONTACT_REPRESENTATION_TOKENS
        )
        self.dataset_max_capacity = dataset_cfg.get("max_capacity", 100_000_000)
        self.dataset_load_mode = dataset_cfg.get("load_mode", "eager")
        self.num_data_workers = int(dataset_cfg.get("num_data_workers", 4))
        self.num_valid_data_workers = self.num_data_workers
        max_total_workers = dataset_cfg.get("max_total_workers")
        if max_total_workers is not None:
            num_valid_loaders = len(dataset_cfg.get("valid_datasets") or {})
            concurrent_loaders = max(self.world_size, 1) + num_valid_loaders
            workers_per_loader = max(0, int(max_total_workers) // concurrent_loaders)
            self.num_data_workers = min(self.num_data_workers, workers_per_loader)
            self.num_valid_data_workers = min(self.num_valid_data_workers, workers_per_loader)
        is_cuda = torch.device(self.device).type == "cuda"
        self.pin_memory = _resolve_auto_bool(
            dataset_cfg.get("pin_memory", "auto"),
            automatic=is_cuda and self.dataset_load_mode == "lazy",
        )
        self.non_blocking_data_transfer = _resolve_auto_bool(
            dataset_cfg.get("non_blocking", "auto"),
            automatic=self.pin_memory,
        )
        self.persistent_workers = _resolve_auto_bool(
            dataset_cfg.get("persistent_workers", "auto"),
            automatic=self.dataset_load_mode == "lazy" and self.num_data_workers > 0,
        )
        self.prefetch_factor = int(dataset_cfg.get("prefetch_factor", 2))
        if self.prefetch_factor <= 0:
            raise ValueError("dataset.prefetch_factor must be positive.")
        if self.is_main_process:
            print_info(
                "Dataset loader: "
                f"mode={self.dataset_load_mode}, train_workers={self.num_data_workers}, "
                f"valid_workers={self.num_valid_data_workers}, pin_memory={self.pin_memory}, "
                f"persistent_workers={self.persistent_workers}, prefetch_factor={self.prefetch_factor}"
            )
        self.train_dataset = None
        self.valid_datasets = {}
        self.collate_fn = None
        self.train_dataset_rank_sharded = False
        dataset_load_start = time.perf_counter()
        self.get_datasets(algo_cfg["dataset"].get("train_dataset_path"), algo_cfg["dataset"].get("valid_datasets"))
        self.dataset_load_seconds_local = time.perf_counter() - dataset_load_start
        self.dataset_load_seconds_by_rank = self._distributed_scalar_values(self.dataset_load_seconds_local)
        self.dataset_load_seconds_mean = sum(self.dataset_load_seconds_by_rank) / len(self.dataset_load_seconds_by_rank)
        self.dataset_load_seconds_max = max(self.dataset_load_seconds_by_rank)

        if cli_cfg["train"]:
            self.num_epochs = int(algo_cfg["num_epochs"])
            self.num_iters_per_epoch = int(algo_cfg.get("num_iters_per_epoch", -1))
            self.start_epoch = 0
            self.best_valid_losses = {}
            self.best_eval_error = np.inf

            dataset_statistics_start = time.perf_counter()
            if (
                algo_cfg.get("update_dataset_statistics", True)
                or self._checkpoint is None
                or not hasattr(self.neural_model, "input_rms")
                or not hasattr(self.neural_model, "output_rms")
                or not hasattr(self, "loss_weights")
            ):
                self.compute_or_sync_dataset_statistics(self.train_dataset)
                self.neural_model_unwrapped.set_input_rms(self.dataset_rms)
                self.neural_model_unwrapped.set_output_rms(self.dataset_rms["prediction_target"])
                self.loss_weights = (
                    1.0 / torch.sqrt(self.dataset_rms["relative_states"].var + 1e-5)
                    if algo_cfg.get("weighted_loss", True)
                    else 1.0
                )
            else:
                print_info("Using dataset statistics from checkpoint...")
            self.dataset_statistics_seconds_local = time.perf_counter() - dataset_statistics_start
            self.dataset_statistics_seconds_by_rank = self._distributed_scalar_values(
                self.dataset_statistics_seconds_local
            )
            self.dataset_statistics_seconds_mean = sum(self.dataset_statistics_seconds_by_rank) / len(
                self.dataset_statistics_seconds_by_rank
            )
            self.dataset_statistics_seconds_max = max(self.dataset_statistics_seconds_by_rank)
            self.mean_predictor_loss_baseline = mean_predictor_loss(
                self.dataset_rms["prediction_target"].var,
                self.loss_weights,
            )

            self._wrap_distributed_model()
            self._init_optimizer(algo_cfg)
            if self._checkpoint is not None and self._checkpoint.get("version", 1) >= 2:
                self._restore_training_state()

            self._checkpoint = None
            self.truncate_grad = algo_cfg.get("truncate_grad", False)
            self.grad_norm = algo_cfg.get("grad_norm", 1.0)
            profiling_cfg = algo_cfg.get("profiling", {})
            self.profile_cuda_synchronize = bool(profiling_cfg.get("cuda_synchronize", False))
            self.profile_cuda_event_timing = bool(profiling_cfg.get("cuda_event_timing", False))
            self.profile_record_samples = bool(profiling_cfg.get("record_samples", False))
            self.profile_warmup_steps = int(profiling_cfg.get("warmup_steps", 0))
            self.profile_phase_steps = int(profiling_cfg.get("phase_steps", 0))
            self.profile_wandb_stats_interval_seconds = profiling_cfg.get("wandb_system_stats_interval_seconds")
            if self.profile_wandb_stats_interval_seconds is not None:
                self.profile_wandb_stats_interval_seconds = float(self.profile_wandb_stats_interval_seconds)
            if self.profile_warmup_steps < 0:
                raise ValueError("profiling.warmup_steps must be non-negative.")
            if self.profile_phase_steps < 0:
                raise ValueError("profiling.phase_steps must be non-negative.")
            if (
                self.profile_wandb_stats_interval_seconds is not None
                and self.profile_wandb_stats_interval_seconds <= 0.0
            ):
                raise ValueError("profiling.wandb_system_stats_interval_seconds must be positive.")
            diagnostics_cfg = algo_cfg.get("diagnostics", {})
            self.diagnostics_enabled = bool(diagnostics_cfg.get("enabled", False))
            self.diagnostic_batches_per_epoch = int(diagnostics_cfg.get("batches_per_epoch", 1))
            if self.diagnostics_enabled and self.diagnostic_batches_per_epoch <= 0:
                raise ValueError("diagnostics.batches_per_epoch must be positive when diagnostics are enabled.")
            self._capture_forward_diagnostics = False
            self._forward_diagnostic_outputs: dict[str, torch.Tensor] = {}
            self._diagnostic_handles: list[Any] = []
            if self.diagnostics_enabled:
                self._register_diagnostic_hooks()
            self._init_logging(cli_cfg)
            if self.is_main_process:
                with open(os.path.join(self.log_dir, "cfg.yaml"), "w") as cfg_file:
                    yaml.dump(cfg, cfg_file)
                if self.logger.wandb:
                    self.logger.log_text_file(os.path.join(self.log_dir, "cfg.yaml"), "training_cfg")

            self._init_evaluator(algo_cfg, cli_cfg)

    @property
    def neural_model_unwrapped(self):
        """Return the underlying model module, unwrapping DDP when active."""
        if isinstance(self.neural_model, DistributedDataParallel):
            return self.neural_model.module
        return self.neural_model

    def _forward_model(self, data: dict[str, torch.Tensor], train: bool, **kwargs):
        """Run model forward, avoiding DDP collectives for rank-local validation."""
        model = self.neural_model if train else self.neural_model_unwrapped
        return model(data, **kwargs)

    def _wrap_distributed_model(self) -> None:
        if not self.is_distributed:
            return
        self.neural_model = DistributedDataParallel(
            self.neural_model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
        )

    def _distributed_mean(self, value: float) -> float:
        if not self.is_distributed:
            return value
        value_tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        value_tensor /= self.world_size
        return float(value_tensor.cpu())

    def _distributed_mean_dict(self, values: dict[str, float]) -> dict[str, float]:
        if not self.is_distributed:
            return values
        return {key: self._distributed_mean(value) for key, value in values.items()}

    def _distributed_scalar_stats(self, value: float) -> tuple[float, float]:
        """Return distributed mean and maximum for one scalar."""
        if not self.is_distributed:
            return value, value
        value_sum = torch.tensor(value, device=self.device, dtype=torch.float64)
        value_max = value_sum.clone()
        dist.all_reduce(value_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(value_max, op=dist.ReduceOp.MAX)
        return float((value_sum / self.world_size).cpu()), float(value_max.cpu())

    def _distributed_scalar_values(self, value: float) -> list[float]:
        """Gather one scalar from every rank in rank order."""
        if not self.is_distributed:
            return [value]
        local_value = torch.tensor(value, device=self.device, dtype=torch.float64)
        gathered = [torch.empty_like(local_value) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_value)
        return [float(rank_value.cpu()) for rank_value in gathered]

    def _capture_peak_gpu_memory(self) -> None:
        """Capture epoch peak allocated and reserved GPU memory across ranks."""
        if not torch.cuda.is_available() or torch.device(self.device).type != "cuda":
            allocated_gib = 0.0
            reserved_gib = 0.0
        else:
            allocated_gib = float(torch.cuda.max_memory_allocated(self.device)) / 1024**3
            reserved_gib = float(torch.cuda.max_memory_reserved(self.device)) / 1024**3
        self.peak_gpu_memory_gib_local = allocated_gib
        (
            self.peak_gpu_memory_gib_mean,
            self.peak_gpu_memory_gib_max,
        ) = self._distributed_scalar_stats(allocated_gib)
        self.peak_gpu_reserved_gib_local = reserved_gib
        (
            self.peak_gpu_reserved_gib_mean,
            self.peak_gpu_reserved_gib_max,
        ) = self._distributed_scalar_stats(reserved_gib)

    def _gather_profile_samples(
        self,
        samples: dict[str, list[float]],
    ) -> tuple[dict[str, list[list[float]]], dict[str, list[float]]]:
        """Gather phase samples once and select each step's slowest-rank path."""
        timer_names = tuple(samples)
        sample_counts = {len(values) for values in samples.values()}
        if len(sample_counts) != 1:
            raise ValueError("Profile timers must have the same number of samples.")
        num_samples = sample_counts.pop()
        if num_samples == 0:
            return {name: [] for name in timer_names}, {name: [] for name in timer_names}
        local_samples = torch.tensor(
            [samples[name] for name in timer_names],
            device=self.device,
            dtype=torch.float64,
        )
        if self.is_distributed:
            gathered = [torch.empty_like(local_samples) for _ in range(self.world_size)]
            dist.all_gather(gathered, local_samples)
        else:
            gathered = [local_samples]
        rank_samples = torch.stack(gathered).cpu()
        step_index = timer_names.index("train_step")
        slowest_ranks = rank_samples[:, step_index, :].argmax(dim=0)
        sample_indices = torch.arange(num_samples)
        by_rank = {name: rank_samples[:, timer_index, :].tolist() for timer_index, name in enumerate(timer_names)}
        critical_path = {
            name: rank_samples[slowest_ranks, timer_index, sample_indices].tolist()
            for timer_index, name in enumerate(timer_names)
        }
        return by_rank, critical_path

    @staticmethod
    def _summarize_samples(samples: list[float]) -> dict[str, float]:
        """Return distribution statistics for a non-empty latency series."""
        if not samples:
            return {}
        values = torch.tensor(samples, dtype=torch.float64)
        mean = float(values.mean())
        std = float(values.std(unbiased=False))
        return {
            "count": float(values.numel()),
            "mean": mean,
            "p50": float(torch.quantile(values, 0.50)),
            "p95": float(torch.quantile(values, 0.95)),
            "min": float(values.min()),
            "max": float(values.max()),
            "cv": std / mean if mean > 0.0 else 0.0,
        }

    def _finalize_train_profile(self) -> None:
        """Resolve train timings and aggregate the DDP critical path before validation."""
        self.time_report.finalize()
        if self.profile_measurement_start_unix is not None:
            self.profile_measurement_end_unix = time.time()
        timing_seconds = self.time_report.as_dict()
        train_total_local = timing_seconds["train_total"]
        self.train_total_seconds_local = train_total_local
        (
            self.train_total_seconds_mean,
            self.train_total_seconds_max,
        ) = self._distributed_scalar_stats(train_total_local)
        measurement_window_local = timing_seconds["train_measurement_window"]
        self.measurement_window_seconds_local = measurement_window_local
        self.measurement_window_seconds_by_rank = self._distributed_scalar_values(measurement_window_local)
        self.measurement_window_seconds_mean = sum(self.measurement_window_seconds_by_rank) / len(
            self.measurement_window_seconds_by_rank
        )
        self.measurement_window_seconds_max = max(self.measurement_window_seconds_by_rank)
        self._capture_peak_gpu_memory()
        self.train_iterator_reset_count_local = float(len(self.train_iterator_reset_steps))
        (
            self.train_iterator_reset_count_mean,
            self.train_iterator_reset_count_max,
        ) = self._distributed_scalar_stats(self.train_iterator_reset_count_local)
        self.profile_samples_by_rank = {}
        self.profile_critical_path_samples = {}
        if not self.profile_record_samples:
            return

        train_step_samples = self.time_report.timers["train_step"].samples
        (
            self.profile_samples_by_rank,
            self.profile_critical_path_samples,
        ) = self._gather_profile_samples({"train_step": train_step_samples})
        phase_start = self.profile_warmup_steps
        phase_end = phase_start + self.profile_phase_steps if self.profile_phase_steps > 0 else len(train_step_samples)
        phase_samples = {
            "train_step": train_step_samples[phase_start:phase_end],
            **{name: self.time_report.timers[name].samples for name in self.profile_phase_timer_names},
        }
        phase_samples_by_rank, phase_critical_path_samples = self._gather_profile_samples(phase_samples)
        self.profile_phase_train_step_samples = phase_critical_path_samples["train_step"]
        self.profile_samples_by_rank.update(
            {name: samples for name, samples in phase_samples_by_rank.items() if name != "train_step"}
        )
        self.profile_critical_path_samples.update(
            {name: samples for name, samples in phase_critical_path_samples.items() if name != "train_step"}
        )
        self.profile_phase_sample_steps = float(phase_end - phase_start)

    def _register_diagnostic_hooks(self) -> None:
        """Capture selected forward representations only on diagnostic batches."""

        def make_hook(name: str):
            def hook(_module, _inputs, output):
                if self._capture_forward_diagnostics:
                    self._forward_diagnostic_outputs[name] = output.detach()

            return hook

        model = self.neural_model_unwrapped
        final_feature_net = getattr(getattr(model, "model", None), "feature_net", None)
        if final_feature_net is not None:
            self._diagnostic_handles.append(final_feature_net.register_forward_hook(make_hook("final_hidden")))
        contact_encoder = getattr(model, "contact_set_encoder", None)
        if contact_encoder is not None:
            self._diagnostic_handles.append(contact_encoder.register_forward_hook(make_hook("contact_representation")))

    def _set_diagnostic_capture(self, enabled: bool) -> None:
        self._capture_forward_diagnostics = enabled
        self._forward_diagnostic_outputs = {}

    def _collect_forward_diagnostics(self, prediction: torch.Tensor) -> dict[str, float]:
        metrics = tensor_distribution_metrics(prediction, "prediction")
        final_hidden = self._forward_diagnostic_outputs.get("final_hidden")
        if final_hidden is not None:
            metrics.update(
                tensor_distribution_metrics(
                    final_hidden,
                    "final_hidden",
                    include_positive_fraction=True,
                )
            )
        contact_representation = self._forward_diagnostic_outputs.get("contact_representation")
        if contact_representation is not None:
            metrics.update(
                tensor_distribution_metrics(
                    contact_representation,
                    "contact_representation",
                )
            )
        return metrics

    def _serialize_dataset_rms(self) -> dict[str, dict[str, Any]]:
        """Return CPU tensors for broadcasting dataset RMS state."""
        return {
            key: {
                "shape": tuple(rms.mean.shape),
                "mean": rms.mean.detach().cpu(),
                "var": rms.var.detach().cpu(),
                "count": rms.count.detach().cpu(),
            }
            for key, rms in self.dataset_rms.items()
        }

    def _load_dataset_rms_state(self, state: dict[str, dict[str, Any]]) -> None:
        """Create dataset RMS modules from broadcast state."""
        self.dataset_rms = {}
        for key, rms_state in state.items():
            rms = RunningMeanStd(shape=tuple(rms_state["shape"]), device=self.device)
            rms.load_state_dict(
                {
                    "mean": rms_state["mean"].to(self.device),
                    "var": rms_state["var"].to(self.device),
                    "count": rms_state["count"].to(self.device),
                }
            )
            self.dataset_rms[key] = rms

    def _restore_training_state(self) -> None:
        checkpoint = self._checkpoint
        if checkpoint is None:
            raise RuntimeError("Cannot restore training state without a checkpoint.")
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print_info("Restored optimizer state from checkpoint.")
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
            print_info(f"Resuming from epoch {self.start_epoch}.")
        if "best_valid_losses" in checkpoint:
            self.best_valid_losses = checkpoint["best_valid_losses"]
        if "best_eval_error" in checkpoint:
            self.best_eval_error = checkpoint["best_eval_error"]
        if "loss_weights" in checkpoint:
            self.loss_weights = checkpoint["loss_weights"].to(self.device)
        if "contact_rms_counts" in checkpoint:
            self.contact_rms_counts = {
                key: value.to(self.device) for key, value in checkpoint["contact_rms_counts"].items()
            }
        print_info("Restored model and training state from checkpoint.")

    def _init_logging(self, cli_cfg: dict[str, Any]) -> None:
        self.log_dir = cli_cfg["logdir"]
        if self.is_main_process and os.path.exists(self.log_dir) and not cli_cfg["skip_check_log_override"]:
            answer = input(f"Logging Directory {self.log_dir} exists, overwrite? [y/n]")
            if answer == "y":
                shutil.rmtree(self.log_dir)
            else:
                raise RuntimeError(f"Refusing to overwrite log directory: {self.log_dir}")
        if self.is_main_process:
            os.makedirs(self.log_dir, exist_ok=True)
        self.model_log_dir = os.path.join(self.log_dir, "nn")
        self.summary_log_dir = os.path.join(self.log_dir, "summaries")
        if self.is_main_process:
            os.makedirs(self.model_log_dir, exist_ok=True)
            os.makedirs(self.summary_log_dir, exist_ok=True)
        if self.is_distributed:
            dist.barrier()

        self.logger = Logger()
        if self.is_main_process:
            self.logger.init_tensorboard(self.summary_log_dir)
        if self.is_main_process and cli_cfg["enable_wandb"]:
            self.logger.init_wandb(
                wandb_project=cli_cfg["wandb_project_name"],
                wandb_name=cli_cfg["wandb_exp_name"],
                wandb_entity=cli_cfg.get("wandb_entity"),
                config=self._wandb_config(),
                save_checkpoints=cli_cfg.get("wandb_save_checkpoints", True),
                system_stats_interval_seconds=self.profile_wandb_stats_interval_seconds,
            )

        self.save_interval = cli_cfg.get("save_interval", 50)
        self.log_interval = cli_cfg.get("log_interval", 1)
        if self.is_main_process:
            Path(self.model_log_dir, "saved_best_eval_model_epochs.txt").touch()
            for valid_dataset_name in self.valid_datasets:
                Path(self.model_log_dir, f"saved_best_valid_{valid_dataset_name}_model_epochs.txt").touch()

    def _wandb_config(self) -> dict[str, Any]:
        """Return a compact config dict for W&B."""
        env_cfg = self.cfg.get("env", {})
        algo_cfg = self.cfg.get("algorithm", {})
        solver_cfg = env_cfg.get("neural_solver_cfg", {})
        inputs_cfg = self.cfg.get("inputs", {})
        contact_cfg = inputs_cfg.get("contact_set", {})
        network_cfg = self.cfg.get("network", {})
        transformer_cfg = network_cfg.get("transformer", {})
        model_cfg = network_cfg.get("model", {})
        return {
            "env_name": env_cfg.get("env_name"),
            "robot_name": env_cfg.get("robot_name"),
            "num_envs": env_cfg.get("num_envs"),
            "contact_mode": solver_cfg.get("contact_mode"),
            "num_contacts_per_env": solver_cfg.get("num_contacts_per_env"),
            "algorithm": algo_cfg.get("name"),
            "seed": self.seed,
            "batch_size": algo_cfg.get("batch_size"),
            "world_size": getattr(self, "world_size", 1),
            "global_batch_windows": int(algo_cfg.get("batch_size", 0)) * getattr(self, "world_size", 1),
            "num_epochs": algo_cfg.get("num_epochs"),
            "num_iters_per_epoch": getattr(self, "num_iters_per_epoch", algo_cfg.get("num_iters_per_epoch")),
            "dataset_max_capacity": self.dataset_max_capacity,
            "dataset_load_seconds_local": getattr(self, "dataset_load_seconds_local", None),
            "dataset_load_seconds_mean": getattr(self, "dataset_load_seconds_mean", None),
            "dataset_load_seconds_max": getattr(self, "dataset_load_seconds_max", None),
            "dataset_statistics_seconds_local": getattr(self, "dataset_statistics_seconds_local", None),
            "dataset_statistics_seconds_mean": getattr(self, "dataset_statistics_seconds_mean", None),
            "dataset_statistics_seconds_max": getattr(self, "dataset_statistics_seconds_max", None),
            "target_mean_loss_baseline": getattr(self, "mean_predictor_loss_baseline", None),
            "lr_start": algo_cfg.get("optimizer", {}).get("lr_start"),
            "lr_end": algo_cfg.get("optimizer", {}).get("lr_end"),
            "lr_schedule": algo_cfg.get("optimizer", {}).get("lr_schedule"),
            "profiling_cuda_synchronize": getattr(self, "profile_cuda_synchronize", False),
            "profiling_cuda_event_timing": getattr(self, "profile_cuda_event_timing", False),
            "profiling_record_samples": getattr(self, "profile_record_samples", False),
            "profiling_warmup_steps": getattr(self, "profile_warmup_steps", 0),
            "profiling_phase_steps": getattr(self, "profile_phase_steps", 0),
            "profiling_wandb_system_stats_interval_seconds": getattr(
                self, "profile_wandb_stats_interval_seconds", None
            ),
            "dataset_load_seconds_by_rank": getattr(self, "dataset_load_seconds_by_rank", []),
            "dataset_statistics_seconds_by_rank": getattr(self, "dataset_statistics_seconds_by_rank", []),
            "diagnostics_enabled": getattr(self, "diagnostics_enabled", False),
            "diagnostic_batches_per_epoch": getattr(self, "diagnostic_batches_per_epoch", 0),
            "model_num_parameters": num_params_torch_model(self.neural_model),
            "contact_encoder_type": contact_cfg.get("encoder_type"),
            "contact_body_latent_dim": contact_cfg.get("body_latent_dim"),
            "contact_hidden_dim": contact_cfg.get("hidden_dim"),
            "transformer_n_layer": transformer_cfg.get("n_layer"),
            "transformer_n_head": transformer_cfg.get("n_head"),
            "transformer_n_embd": transformer_cfg.get("n_embd"),
            "model_mlp_layer_sizes": model_cfg.get("mlp", {}).get("layer_sizes"),
        }

    def _init_evaluator(self, algo_cfg: dict[str, Any], cli_cfg: dict[str, Any]) -> None:
        eval_cfg = algo_cfg.get("eval", {})
        cli_eval_interval = cli_cfg.get("eval_interval")
        self.eval_interval = eval_cfg.get("interval", 0) if cli_eval_interval is None else cli_eval_interval
        self.evaluator = None
        if self.eval_interval <= 0 or not self.is_main_process:
            return

        self.eval_mode = eval_cfg.get("mode", "dataset")
        self.action_mode = eval_cfg.get("action_mode", "joint_f")
        self.eval_horizon = int(eval_cfg.get("rollout_horizon", 5))
        self.num_eval_rollouts = int(eval_cfg.get("num_rollouts", self.neural_env.num_envs))
        self.eval_dataset_path = eval_cfg.get("dataset_path")
        self.eval_passive = bool(eval_cfg.get("passive", True))
        self.eval_require_terrain_context = bool(eval_cfg.get("require_terrain_context", False))
        self.eval_contact_context_validation = eval_cfg.get("contact_context_validation")
        self.eval_state_context_tolerance = float(eval_cfg.get("state_context_tolerance", 1.0e-5))
        self.eval_contact_context_tolerance = float(eval_cfg.get("contact_context_tolerance", 1.0e-4))
        self.eval_contact_context_normal_tolerance = float(eval_cfg.get("contact_context_normal_tolerance", 1.0e-3))
        self.eval_contact_context_velocity_tolerance = float(eval_cfg.get("contact_context_velocity_tolerance", 1.0e-3))
        self.eval_render = bool(cli_cfg.get("render", False))

        if self.action_mode not in ("action", "joint_f"):
            raise NotImplementedError(f"Training rollout eval does not support action_mode={self.action_mode!r}.")
        if self.eval_mode not in ("sample", "dataset", "env"):
            raise NotImplementedError(f"Training rollout eval does not support eval.mode={self.eval_mode!r}.")
        if self.eval_mode == "dataset" and self.eval_dataset_path is None:
            raise ValueError("algorithm.eval.dataset_path is required when eval.mode='dataset'.")

        self.evaluator = TrainingRolloutEvaluator(
            self.neural_env,
            hdf5_dataset_path=self.eval_dataset_path,
            eval_horizon=self.eval_horizon,
            dataset_contact_representation=self.dataset_contact_representation,
            require_solver_active=self.require_solver_active,
            device=self.device,
            require_terrain_context=self.eval_require_terrain_context,
            contact_context_validation=self.eval_contact_context_validation,
            state_context_tolerance=self.eval_state_context_tolerance,
            contact_context_tolerance=self.eval_contact_context_tolerance,
            contact_context_normal_tolerance=self.eval_contact_context_normal_tolerance,
            contact_context_velocity_tolerance=self.eval_contact_context_velocity_tolerance,
        )

    def _init_optimizer(self, algo_cfg: dict[str, Any]) -> None:
        loss_cfg = algo_cfg.get("loss", {})
        loss_type = loss_cfg.get("type", "MSE").lower()
        if loss_type == "mse":
            self.loss_func = torch.nn.MSELoss()
        elif loss_type == "l1":
            self.loss_func = torch.nn.L1Loss()
        elif loss_type == "smoothl1":
            self.loss_func = torch.nn.SmoothL1Loss(beta=loss_cfg.get("beta", 0.1))
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

        self.lr_start = float(algo_cfg["optimizer"]["lr_start"])
        self.lr_end = float(algo_cfg["optimizer"].get("lr_end", 0.0))
        self.lr_schedule = algo_cfg["optimizer"]["lr_schedule"]
        self.betas = tuple(algo_cfg["optimizer"].get("betas", (0.9, 0.999)))
        optimizer_type = algo_cfg["optimizer"].get("type", "adam").lower()

        if optimizer_type == "adamw":
            weight_decay = float(algo_cfg["optimizer"].get("weight_decay", 0.1))
            param_dict = {name: param for name, param in self.neural_model.named_parameters() if param.requires_grad}
            optim_groups = [
                {"params": [param for param in param_dict.values() if param.dim() >= 2], "weight_decay": weight_decay},
                {"params": [param for param in param_dict.values() if param.dim() < 2], "weight_decay": 0.0},
            ]
            fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
            extra_args = {"fused": True} if fused_available and "cuda" in str(self.device) else {}
            self.optimizer = torch.optim.AdamW(optim_groups, lr=self.lr_start, betas=self.betas, **extra_args)
        else:
            self.optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=self.lr_start, betas=self.betas)

    def get_datasets(self, train_dataset_path, valid_datasets_cfg) -> None:
        self.train_dataset = create_batch_transition_dataset(
            load_mode=self.dataset_load_mode,
            batch_size=self.batch_size,
            hdf5_dataset_path=train_dataset_path,
            max_capacity=self.dataset_max_capacity,
            device=self.device,
        )
        if valid_datasets_cfg is not None and (not self.is_distributed or self.is_main_process):
            for valid_dataset_name, valid_dataset_path in valid_datasets_cfg.items():
                self.valid_datasets[valid_dataset_name] = create_batch_transition_dataset(
                    load_mode=self.dataset_load_mode,
                    batch_size=self.batch_size,
                    hdf5_dataset_path=valid_dataset_path,
                    device=self.device,
                )
        self.batch_size = 1
        self.collate_fn = collate_fn_BatchTransitionDataset

    def _make_dataloader(
        self,
        *,
        dataset,
        batch_size: int,
        shuffle: bool,
        sampler=None,
        drop_last: bool,
        persistent_workers: bool | None = None,
        num_workers: int | None = None,
    ) -> DataLoader:
        """Create a consistently configured training or validation DataLoader."""
        loader_workers = self.num_data_workers if num_workers is None else num_workers
        use_persistent_workers = (
            self.persistent_workers if persistent_workers is None else persistent_workers
        ) and loader_workers > 0
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "collate_fn": self.collate_fn,
            "shuffle": shuffle,
            "sampler": sampler,
            "num_workers": loader_workers,
            "drop_last": drop_last,
            "pin_memory": self.pin_memory,
            "persistent_workers": use_persistent_workers,
        }
        if loader_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["worker_init_fn"] = _data_worker_init
        return DataLoader(**kwargs)

    def compute_dataset_statistics(self, dataset) -> None:
        dataloader = self._make_dataloader(
            dataset=dataset,
            batch_size=max(512, self.batch_size),
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
        )
        self.dataset_rms = {}
        self.contact_rms_counts: dict[str, torch.Tensor] = {}
        contact_moments: dict[str, MaskedContactMoments] = {}
        contact_token_moments: ContactTokenMoments | None = None
        for data in tqdm(dataloader, disable=not self.is_main_process):
            data = self.preprocess_data_batch(data)
            data["relative_states"] = self.neural_solver.convert_next_states_to_prediction(
                states=data["states"],
                next_states=data["next_states"],
                dt=self.neural_env.frame_dt,
                prediction_type="relative",
            )
            contact_masks = data.get("contact_masks")
            for key, value in data.items():
                if value is None or key in {"contact_masks", "contact_token_overflow"}:
                    continue
                if key == "contact_tokens":
                    if contact_token_moments is None:
                        categorical_channels = (
                            ACTIVE15_CATEGORICAL_CHANNELS
                            if self.neural_solver.contact_representation
                            in {CONTACT_REPRESENTATION_RAW15, CONTACT_REPRESENTATION_ACTIVE15}
                            else None
                        )
                        contact_token_moments = ContactTokenMoments(
                            value.shape[-1],
                            self.device,
                            categorical_channels=categorical_channels,
                        )
                    contact_token_moments.update(value)
                    continue
                use_masked_contact_rms = self.neural_solver.contact_mode == "newton_native" and key.startswith(
                    "contact_"
                )
                if use_masked_contact_rms and contact_masks is None:
                    raise ValueError("Newton-native contact RMS requires explicit contact_masks.")
                if use_masked_contact_rms:
                    assert contact_masks is not None
                    if key not in contact_moments:
                        contact_moments[key] = MaskedContactMoments.from_batch(value, contact_masks)
                    contact_moments[key].update(value, contact_masks)
                    continue
                if key not in self.dataset_rms:
                    rms_shape = value.shape[2:]
                    self.dataset_rms[key] = RunningMeanStd(shape=rms_shape, device=self.device)
                self.dataset_rms[key].update(value, batch_dim=True, time_dim=True)
        if self.is_distributed and self.train_dataset_rank_sharded:
            for rms in self.dataset_rms.values():
                _synchronize_running_mean_std(rms)
            for moments in contact_moments.values():
                moments.synchronize()
            if contact_token_moments is not None:
                contact_token_moments.synchronize()
        for key, moments in contact_moments.items():
            rms, counts = moments.finalize(self.device)
            self.dataset_rms[key] = rms
            self.contact_rms_counts[key] = counts
            sparse_slots = int(((counts > 0) & (counts < MIN_CONTACT_RMS_SAMPLES)).sum())
            if self.is_main_process:
                print_info(
                    f"Contact RMS {key}: min_count={int(counts.min())}, max_count={int(counts.max())}, "
                    f"sparse_slots={sparse_slots}/{counts.numel()} "
                    f"(pooled fallback below {MIN_CONTACT_RMS_SAMPLES} samples)."
                )
        if contact_token_moments is not None:
            self.dataset_rms["contact_tokens"] = contact_token_moments.finalize(self.device)
            if self.is_main_process:
                print_info(
                    f"Contact token RMS: samples={int(self.dataset_rms['contact_tokens'].count.item())}, "
                    f"dim={self.dataset_rms['contact_tokens'].mean.numel()}"
                )

    def compute_or_sync_dataset_statistics(self, dataset) -> None:
        """Compute dataset statistics once, then share them with DDP ranks."""
        if self.is_distributed and self.train_dataset_rank_sharded:
            if self.is_main_process:
                print_info("Computing distributed dataset statistics...")
            self.compute_dataset_statistics(dataset)
            if self.is_main_process:
                print_info("Finished computing distributed dataset statistics...")
            return

        rms_state: list[dict[str, dict[str, Any]] | None] = [None]
        if self.is_main_process:
            print_info("Computing dataset statistics...")
            self.compute_dataset_statistics(dataset)
            print_info("Finished computing dataset statistics...")
            rms_state[0] = self._serialize_dataset_rms()

        if self.is_distributed:
            dist.broadcast_object_list(rms_state, src=0)
            if not self.is_main_process:
                if rms_state[0] is None:
                    raise RuntimeError("Failed to receive dataset statistics from rank 0.")
                self._load_dataset_rms_state(rms_state[0])

    def get_scheduled_learning_rate(self, iteration: int, total_iterations: int) -> float:
        if self.lr_schedule == "constant":
            return self.lr_start
        if self.lr_schedule == "linear":
            ratio = iteration / total_iterations
            return self.lr_start * (1.0 - ratio) + self.lr_end * ratio
        if self.lr_schedule == "cosine":
            decay_ratio = iteration / total_iterations
            return self.lr_end + 0.5 * (1.0 + np.cos(np.pi * decay_ratio)) * (self.lr_start - self.lr_end)
        raise NotImplementedError(f"Unsupported lr schedule: {self.lr_schedule}")

    @torch.no_grad()
    def transfer_data_batch(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remove metadata fields and transfer a batch to the training device."""
        for key in NON_MODEL_DATA_KEYS.intersection(data):
            data.pop(key)
        for key, value in data.items():
            data[key] = value.to(self.device, non_blocking=self.non_blocking_data_transfer)
        return data

    @torch.no_grad()
    def preprocess_transferred_data_batch(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Construct model inputs and targets after device transfer."""
        if "contact_masks" in data:
            data["contact_masks"] = data["contact_masks"].bool()
        elif "contact_tokens" not in data:
            data["contact_masks"] = self.neural_solver.get_contact_masks(
                data["contact_depths"],
                data["contact_thicknesses_0"],
                data["contact_thicknesses_1"],
            )
        self.neural_solver.process_neural_model_inputs(data)
        data["prediction_target"] = self.neural_solver.convert_next_states_to_prediction(
            states=data["states"],
            next_states=data["next_states"],
            dt=self.neural_env.frame_dt,
        )
        return data

    @torch.no_grad()
    def preprocess_data_batch(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Transfer and preprocess one data batch."""
        data = self.transfer_data_batch(data)
        return self.preprocess_transferred_data_batch(data)

    def compute_loss(self, data: dict[str, torch.Tensor], train: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if self.neural_model_unwrapped.is_rnn:
            self.neural_model_unwrapped.init_rnn(self.batch_size)
        prediction_target = data["prediction_target"]
        prediction = self._forward_model(data, train)
        loss = self.loss_func(prediction * self.loss_weights, prediction_target * self.loss_weights)
        return loss, prediction

    @torch.no_grad()
    def compute_state_error_metrics(
        self,
        data: dict[str, torch.Tensor],
        prediction: torch.Tensor,
    ) -> dict[str, float]:
        """Convert predictions to physical next states and return unweighted errors."""
        predicted_next_states = self.neural_solver.convert_prediction_to_next_states(
            states=data["states"],
            prediction=prediction,
            dt=self.neural_env.frame_dt,
        )
        self.neural_solver.wrap2PI(predicted_next_states)
        return state_error_metrics(
            predicted_next_states,
            data["next_states"],
            dof_q=self.neural_solver.dof_q_per_env,
        )

    def one_epoch(
        self,
        train: bool,
        dataloader,
        dataloader_iter,
        num_batches: int,
        shuffle: bool = False,
        distributed_reduce: bool = True,
    ):
        self.neural_model.train(train)
        phase = "train" if train else "validation"
        sum_loss = torch.tensor(0.0, device=self.device)
        sum_loss_itemized: dict[str, float] = {}
        diagnostic_info: dict[str, float] = {}
        diagnostic_batch_count = 0
        grad_info = {"grad_norm_before_clip": 0.0} if train else {}
        if train and self.truncate_grad:
            grad_info["grad_norm_after_clip"] = 0.0
            grad_info["clip_fraction"] = 0.0

        iterator_reset_steps: list[int] = []
        phase_sample_start = getattr(self, "profile_warmup_steps", 0)
        phase_sample_end = (
            phase_sample_start + getattr(self, "profile_phase_steps", 0)
            if getattr(self, "profile_phase_steps", 0) > 0
            else num_batches
        )
        if train and getattr(self, "profile_record_samples", False):
            for timer_name in self.profile_phase_timer_names:
                self.time_report.set_timer_enabled(timer_name, False)
        with torch.set_grad_enabled(train):
            for batch_index in tqdm(range(num_batches), disable=not self.is_main_process):
                if train:
                    if getattr(self, "profile_record_samples", False):
                        if batch_index == phase_sample_start:
                            for timer_name in self.profile_phase_timer_names:
                                self.time_report.set_timer_enabled(timer_name, True)
                        elif batch_index == phase_sample_end:
                            for timer_name in self.profile_phase_timer_names:
                                self.time_report.set_timer_enabled(timer_name, False)
                        if batch_index == self.profile_warmup_steps:
                            self.profile_measurement_start_unix = time.time()
                            self.time_report.start_timer("train_measurement_window")
                    self.time_report.start_timer("train_step")
                capture_diagnostics = (
                    train and self.diagnostics_enabled and batch_index < self.diagnostic_batches_per_epoch
                )
                self._set_diagnostic_capture(capture_diagnostics)

                with TimeProfiler(self.time_report, f"{phase}_dataloader"):
                    try:
                        data = next(dataloader_iter)
                    except StopIteration:
                        iterator_reset_steps.append(batch_index)
                        if shuffle and self.train_dataset is not None:
                            self.train_dataset.shuffle()
                        dataloader_iter = iter(dataloader)
                        data = next(dataloader_iter)

                with TimeProfiler(self.time_report, f"{phase}_data_transfer"):
                    data = self.transfer_data_batch(data)
                with TimeProfiler(self.time_report, f"{phase}_preprocess"):
                    data = self.preprocess_transferred_data_batch(data)

                if train:
                    with TimeProfiler(self.time_report, "train_zero_grad"):
                        self.optimizer.zero_grad()
                with TimeProfiler(self.time_report, f"{phase}_forward_loss"):
                    loss, prediction = self.compute_loss(data, train)
                with TimeProfiler(self.time_report, f"{phase}_metrics"):
                    loss_itemized = self.compute_state_error_metrics(data, prediction)
                    if capture_diagnostics:
                        batch_diagnostics = self._collect_forward_diagnostics(prediction)
                        for key, value in batch_diagnostics.items():
                            diagnostic_info[key] = diagnostic_info.get(key, 0.0) + value
                        diagnostic_batch_count += 1
                self._set_diagnostic_capture(False)

                if train:
                    with TimeProfiler(self.time_report, "train_backward_ddp"):
                        loss.backward()
                    with TimeProfiler(self.time_report, "train_gradient_diagnostics"):
                        with torch.no_grad():
                            grad_norm_before_clip = float(grad_norm(self.neural_model.parameters()).detach().cpu())
                            grad_info["grad_norm_before_clip"] += grad_norm_before_clip
                            if self.truncate_grad:
                                grad_info["clip_fraction"] += float(grad_norm_before_clip > self.grad_norm)
                                clip_grad_norm_(self.neural_model.parameters(), self.grad_norm)
                                grad_info["grad_norm_after_clip"] += float(
                                    grad_norm(self.neural_model.parameters()).detach().cpu()
                                )
                    with TimeProfiler(self.time_report, "train_optimizer_step"):
                        self.optimizer.step()

                with TimeProfiler(self.time_report, f"{phase}_bookkeeping"):
                    sum_loss += loss.detach()
                    for key, value in loss_itemized.items():
                        sum_loss_itemized[key] = sum_loss_itemized.get(key, 0.0) + value
                if train:
                    self.time_report.end_timer("train_step")

        if train:
            if getattr(self, "profile_record_samples", False):
                self.time_report.end_timer("train_measurement_window")
            self.train_iterator_reset_steps = iterator_reset_steps
        self._set_diagnostic_capture(False)
        avg_loss = sum_loss.cpu().item() / num_batches
        avg_loss_itemized = {key: value / num_batches for key, value in sum_loss_itemized.items()}
        if train:
            for key in grad_info:
                grad_info[key] /= num_batches
        if diagnostic_batch_count > 0:
            diagnostic_info = {key: value / diagnostic_batch_count for key, value in diagnostic_info.items()}
        if distributed_reduce:
            avg_loss = self._distributed_mean(avg_loss)
            avg_loss_itemized = self._distributed_mean_dict(avg_loss_itemized)
            grad_info = self._distributed_mean_dict(grad_info)
            diagnostic_info = self._distributed_mean_dict(diagnostic_info)
        return avg_loss, avg_loss_itemized, grad_info, diagnostic_info

    def train(self) -> None:
        if self.train_dataset is None:
            raise RuntimeError("Training dataset has not been initialized.")
        train_sampler = (
            DistributedSampler(
                cast(Any, self.train_dataset),
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            if self.is_distributed and not self.train_dataset_rank_sharded
            else None
        )
        train_loader = self._make_dataloader(
            dataset=cast(Any, self.train_dataset),
            batch_size=self.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
        )
        self.num_train_batches = len(train_loader) if self.num_iters_per_epoch == -1 else self.num_iters_per_epoch

        valid_loaders = {}
        valid_loader_iters = {}
        if self.is_main_process:
            for valid_dataset_name, valid_dataset in self.valid_datasets.items():
                valid_loaders[valid_dataset_name] = self._make_dataloader(
                    dataset=valid_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=self.num_valid_data_workers,
                )
                self.best_valid_losses.setdefault(valid_dataset_name, np.inf)

        if self.profile_record_samples and self.profile_warmup_steps >= self.num_train_batches:
            raise ValueError("profiling.warmup_steps must be smaller than the number of training iterations.")
        if (
            self.profile_record_samples
            and self.profile_phase_steps > 0
            and self.profile_warmup_steps + self.profile_phase_steps > self.num_train_batches
        ):
            raise ValueError("profiling warmup_steps + phase_steps must not exceed the training iterations.")
        self.time_report = TimeReport(
            cuda_synchronize=self.profile_cuda_synchronize,
            cuda_event_timing=self.profile_cuda_event_timing,
            record_samples=self.profile_record_samples,
        )
        batch_sections = (
            "dataloader",
            "data_transfer",
            "preprocess",
            "forward_loss",
            "metrics",
            "bookkeeping",
        )
        self.time_report.add_timers(
            [
                "epoch",
                "train_total",
                "train_iterator_setup",
                "train_measurement_window",
                "train_step",
                "train_backward_ddp",
                "train_gradient_diagnostics",
                "train_zero_grad",
                "train_optimizer_step",
                "validation_total",
                "validation_iterator_setup",
                "rollout_eval",
                *(f"train_{section}" for section in batch_sections),
                *(f"validation_{section}" for section in batch_sections),
            ]
        )
        self.profile_phase_timer_names = (
            "train_dataloader",
            "train_data_transfer",
            "train_preprocess",
            "train_zero_grad",
            "train_forward_loss",
            "train_metrics",
            "train_backward_ddp",
            "train_gradient_diagnostics",
            "train_optimizer_step",
            "train_bookkeeping",
        )
        self.profile_step_timer_names = ("train_step", *self.profile_phase_timer_names)
        self.global_batch_windows = self.configured_batch_size * self.world_size
        self.optimizer_steps = self.start_epoch * self.num_train_batches
        self.windows_seen = self.optimizer_steps * self.global_batch_windows
        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self.time_report.reset_timer()
            with TimeProfiler(self.time_report, "epoch"):
                self.lr = self.get_scheduled_learning_rate(epoch, self.num_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
                self.logger.init_epoch(epoch)

                self.profile_measurement_start_unix = None
                self.profile_measurement_end_unix = None
                if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
                    torch.cuda.synchronize(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                parameter_snapshot = (
                    snapshot_trainable_parameters(self.neural_model_unwrapped)
                    if self.is_main_process and self.diagnostics_enabled
                    else None
                )
                with TimeProfiler(self.time_report, "train_total"):
                    with TimeProfiler(self.time_report, "train_iterator_setup"):
                        train_loader_iter = iter(train_loader)
                    (
                        avg_train_loss,
                        avg_train_loss_itemized,
                        grad_info,
                        diagnostic_info,
                    ) = self.one_epoch(
                        train=True,
                        dataloader=train_loader,
                        dataloader_iter=train_loader_iter,
                        num_batches=self.num_train_batches,
                        shuffle=True,
                    )
                self._finalize_train_profile()
                if parameter_snapshot is not None:
                    diagnostic_info.update(
                        parameter_change_metrics(
                            parameter_snapshot,
                            self.neural_model_unwrapped,
                        )
                    )
                avg_valid_losses, avg_valid_losses_itemized = {}, {}
                main_process_exception = None
                main_process_traceback = None
                if self.is_main_process:
                    try:
                        with TimeProfiler(self.time_report, "validation_total"):
                            with TimeProfiler(self.time_report, "validation_iterator_setup"):
                                valid_loader_iters = _reset_validation_iterators(valid_loaders)
                            for valid_dataset_name in self.valid_datasets:
                                (
                                    avg_valid_losses[valid_dataset_name],
                                    avg_valid_losses_itemized[valid_dataset_name],
                                    _,
                                    _,
                                ) = self.one_epoch(
                                    train=False,
                                    dataloader=valid_loaders[valid_dataset_name],
                                    dataloader_iter=valid_loader_iters[valid_dataset_name],
                                    num_batches=min(self.num_valid_batches, len(valid_loaders[valid_dataset_name])),
                                    distributed_reduce=False,
                                )
                        with TimeProfiler(self.time_report, "rollout_eval"):
                            if self.eval_interval > 0 and (epoch + 1) % self.eval_interval == 0:
                                self.eval(epoch)
                    except Exception as exc:
                        main_process_exception = exc
                        main_process_traceback = traceback.format_exc()
                if self.is_distributed:
                    error_payload = [main_process_traceback]
                    dist.broadcast_object_list(error_payload, src=0)
                    main_process_traceback = error_payload[0]
                if main_process_traceback is not None:
                    message = f"Main-process validation or rollout evaluation failed:\n{main_process_traceback}"
                    if main_process_exception is not None:
                        raise RuntimeError(message) from main_process_exception
                    raise RuntimeError(message)

            self.time_report.finalize()
            self.optimizer_steps += self.num_train_batches
            self.windows_seen += self.num_train_batches * self.global_batch_windows
            if self.is_main_process and epoch % self.log_interval == 0:
                self._log_training_epoch_summary(
                    epoch,
                    avg_train_loss,
                    avg_train_loss_itemized,
                    avg_valid_losses,
                    avg_valid_losses_itemized,
                    grad_info,
                    diagnostic_info,
                )
            if self.is_main_process:
                self.logger.flush()
            if self.is_main_process and self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
                self.save_model(f"model_epoch{epoch}")
            for valid_dataset_name in self.valid_datasets:
                if (
                    self.is_main_process
                    and avg_valid_losses[valid_dataset_name] < self.best_valid_losses[valid_dataset_name]
                ):
                    self.best_valid_losses[valid_dataset_name] = avg_valid_losses[valid_dataset_name]
                    self.save_model(f"best_valid_{valid_dataset_name}_model")
                    with open(
                        os.path.join(self.model_log_dir, f"saved_best_valid_{valid_dataset_name}_model_epochs.txt"), "a"
                    ) as fp:
                        fp.write(f"{epoch}\n")
                    print_ok(
                        f"Save Best Valid {valid_dataset_name} Model at Epoch {epoch} "
                        f"with loss {avg_valid_losses[valid_dataset_name]:.8f}."
                    )

        for handle in self._diagnostic_handles:
            handle.remove()
        if self.is_main_process:
            self.save_model("final_model")
            self.logger.finish()

    def _log_profile_metrics(self, epoch: int) -> None:
        """Log inclusive and steady-state distributed profiling metrics."""
        sequence_length = getattr(self, "sample_sequence_length", 1)
        for suffix, value in (
            ("local", self.train_total_seconds_local),
            ("mean", self.train_total_seconds_mean),
            ("max", self.train_total_seconds_max),
        ):
            self.logger.add_scalar(f"performance/train_total_seconds_{suffix}/epoch", value, epoch)
        if self.train_total_seconds_max > 0.0:
            inclusive_steps_per_second = self.num_train_batches / self.train_total_seconds_max
            self.logger.add_scalar(
                "performance/inclusive_optimizer_steps_per_second/epoch",
                inclusive_steps_per_second,
                epoch,
            )
            self.logger.add_scalar(
                "performance/inclusive_windows_per_second/epoch",
                inclusive_steps_per_second * self.global_batch_windows,
                epoch,
            )
            self.logger.add_scalar(
                "performance/inclusive_processed_sequence_positions_per_second/epoch",
                inclusive_steps_per_second * self.global_batch_windows * sequence_length,
                epoch,
            )

        for suffix, value in (
            ("local", self.measurement_window_seconds_local),
            ("mean", self.measurement_window_seconds_mean),
            ("max", self.measurement_window_seconds_max),
        ):
            self.logger.add_scalar(
                f"performance/measurement_window_seconds_{suffix}/epoch",
                value,
                epoch,
            )
        for rank, value in enumerate(self.measurement_window_seconds_by_rank):
            self.logger.add_scalar(
                f"performance/measurement_window_seconds_rank_{rank}/epoch",
                value,
                epoch,
            )

        for name, value in (
            ("allocated_local", self.peak_gpu_memory_gib_local),
            ("allocated_mean", self.peak_gpu_memory_gib_mean),
            ("allocated_max", self.peak_gpu_memory_gib_max),
            ("reserved_local", self.peak_gpu_reserved_gib_local),
            ("reserved_mean", self.peak_gpu_reserved_gib_mean),
            ("reserved_max", self.peak_gpu_reserved_gib_max),
        ):
            self.logger.add_scalar(f"memory/peak_gpu_{name}_gib/epoch", value, epoch)
        self.logger.add_scalar(
            "performance/train_iterator_resets_local/epoch",
            self.train_iterator_reset_count_local,
            epoch,
        )
        self.logger.add_scalar(
            "performance/train_iterator_resets_max/epoch",
            self.train_iterator_reset_count_max,
            epoch,
        )
        if self.profile_measurement_start_unix is not None:
            self.logger.add_scalar(
                "performance/measurement_start_unix/epoch",
                self.profile_measurement_start_unix,
                epoch,
            )
        if self.profile_measurement_end_unix is not None:
            self.logger.add_scalar(
                "performance/measurement_end_unix/epoch",
                self.profile_measurement_end_unix,
                epoch,
            )

        if not self.profile_critical_path_samples:
            return
        warmup_steps = self.profile_warmup_steps
        step_samples = self.profile_critical_path_samples["train_step"]
        measured_steps = step_samples[warmup_steps:]
        latency_start = warmup_steps + self.profile_phase_steps if self.profile_phase_steps > 0 else warmup_steps
        latency_samples = step_samples[latency_start:]
        step_summary = self._summarize_samples(latency_samples)
        measured_summary = self._summarize_samples(measured_steps)
        phase_step_summary = self._summarize_samples(self.profile_phase_train_step_samples)
        if not step_summary or not measured_summary:
            return
        for label, summary in (
            ("steady", step_summary),
            ("all_measured", measured_summary),
            ("phase_sample", phase_step_summary),
        ):
            for statistic in ("mean", "p50", "p95", "min", "max"):
                if statistic in summary:
                    self.logger.add_scalar(
                        f"performance/{label}_train_step_ms_{statistic}/epoch",
                        summary[statistic] * 1000.0,
                        epoch,
                    )
        self.logger.add_scalar(
            "performance/steady_train_step_cv/epoch",
            step_summary["cv"],
            epoch,
        )
        self.logger.add_scalar(
            "performance/measurement_steps/epoch",
            measured_summary["count"],
            epoch,
        )
        self.logger.add_scalar(
            "performance/latency_sample_steps/epoch",
            step_summary["count"],
            epoch,
        )
        self.logger.add_scalar(
            "performance/phase_sample_steps/epoch",
            getattr(self, "profile_phase_sample_steps", 0.0),
            epoch,
        )
        if step_summary["mean"] > 0.0 and phase_step_summary:
            self.logger.add_scalar(
                "performance/phase_vs_uninstrumented_step_mean_ratio/epoch",
                phase_step_summary["mean"] / step_summary["mean"],
                epoch,
            )
        if self.measurement_window_seconds_max > 0.0:
            measured_steps_per_second = len(measured_steps) / self.measurement_window_seconds_max
            self.logger.add_scalar(
                "performance/steady_optimizer_steps_per_second/epoch",
                measured_steps_per_second,
                epoch,
            )
            self.logger.add_scalar(
                "performance/steady_windows_per_second/epoch",
                measured_steps_per_second * self.global_batch_windows,
                epoch,
            )
            self.logger.add_scalar(
                "performance/steady_processed_sequence_positions_per_second/epoch",
                measured_steps_per_second * self.global_batch_windows * sequence_length,
                epoch,
            )

        for rank, rank_samples in enumerate(self.profile_samples_by_rank["train_step"]):
            rank_summary = self._summarize_samples(rank_samples[latency_start:])
            for statistic in ("mean", "p50", "p95"):
                self.logger.add_scalar(
                    f"performance/rank_{rank}_train_step_ms_{statistic}/epoch",
                    rank_summary[statistic] * 1000.0,
                    epoch,
                )

        phase_mean_seconds = 0.0
        for timer_name, timer_samples in self.profile_critical_path_samples.items():
            if timer_name == "train_step":
                continue
            timer_summary = self._summarize_samples(timer_samples)
            phase_mean_seconds += timer_summary["mean"]
            for statistic in ("mean", "p50", "p95"):
                self.logger.add_scalar(
                    f"timing/critical_path_{timer_name}_ms_{statistic}/epoch",
                    timer_summary[statistic] * 1000.0,
                    epoch,
                )
            if phase_step_summary.get("mean", 0.0) > 0.0:
                self.logger.add_scalar(
                    f"timing/critical_path_{timer_name}_fraction/epoch",
                    timer_summary["mean"] / phase_step_summary["mean"],
                    epoch,
                )
        if phase_step_summary:
            self.logger.add_scalar(
                "timing/critical_path_phase_unattributed_ms/epoch",
                (phase_step_summary["mean"] - phase_mean_seconds) * 1000.0,
                epoch,
            )

        reset_indices = {
            index for index in self.train_iterator_reset_steps if warmup_steps <= index < len(step_samples)
        }
        reset_samples = [step_samples[index] for index in sorted(reset_indices)]
        non_reset_samples = [
            value for index, value in enumerate(step_samples) if index >= latency_start and index not in reset_indices
        ]
        for label, samples in (("reset", reset_samples), ("non_reset", non_reset_samples)):
            summary = self._summarize_samples(samples)
            for statistic in ("mean", "p50", "p95", "max"):
                if statistic in summary:
                    self.logger.add_scalar(
                        f"performance/{label}_train_step_ms_{statistic}/epoch",
                        summary[statistic] * 1000.0,
                        epoch,
                    )

        print_info(
            "[Profile] slowest-rank steady step: "
            f"mean={step_summary['mean'] * 1000.0:.3f} ms, "
            f"p50={step_summary['p50'] * 1000.0:.3f} ms, "
            f"p95={step_summary['p95'] * 1000.0:.3f} ms, "
            f"iterator_resets={int(self.train_iterator_reset_count_max)}"
        )

    def _log_training_epoch_summary(
        self,
        epoch: int,
        avg_train_loss: float,
        avg_train_loss_itemized: dict[str, float],
        avg_valid_losses: dict[str, float],
        avg_valid_losses_itemized: dict[str, dict[str, float]],
        grad_info: dict[str, float],
        diagnostic_info: dict[str, float],
    ) -> None:
        print_info("-" * 100)
        print_info(f"Epoch {epoch}")
        print_info(f"[Train] loss = {avg_train_loss:.8f}, itemized = {format_dict(avg_train_loss_itemized, 8)}")
        for valid_dataset_name in self.valid_datasets:
            print_info(
                f"[Valid] dataset [{valid_dataset_name}]: loss = {avg_valid_losses[valid_dataset_name]:.8f}, "
                f"itemized = {format_dict(avg_valid_losses_itemized[valid_dataset_name], 8)}"
            )
        print_info(f"[Time Report] {self.time_report.print(string_mode=True, in_second=True)}")
        print_info(f"[Grad Info] {format_dict(grad_info, 3)}")
        if diagnostic_info:
            print_info(f"[Diagnostics] {format_dict(diagnostic_info, 6)}")
        self.logger.add_scalar("params/lr/epoch", self.lr, epoch)
        self.logger.add_scalar("training/train_loss/epoch", avg_train_loss, epoch)
        if "grad_norm_before_clip" in grad_info:
            self.logger.add_scalar("training/gradients_before_clip/epoch", grad_info["grad_norm_before_clip"], epoch)
        if "grad_norm_after_clip" in grad_info:
            self.logger.add_scalar("training/gradients_after_clip/epoch", grad_info["grad_norm_after_clip"], epoch)
        if "clip_fraction" in grad_info:
            self.logger.add_scalar("training/gradient_clip_fraction/epoch", grad_info["clip_fraction"], epoch)
        for key, value in diagnostic_info.items():
            self.logger.add_scalar(f"diagnostics/{key}/epoch", value, epoch)
        self.logger.add_scalar(
            "diagnostics/target_mean_loss_baseline/epoch",
            self.mean_predictor_loss_baseline,
            epoch,
        )
        self.logger.add_scalar("progress/optimizer_steps/epoch", self.optimizer_steps, epoch)
        self.logger.add_scalar("progress/windows_seen/epoch", self.windows_seen, epoch)
        timing_seconds = self.time_report.as_dict()
        sampled_phase_timers = set(getattr(self, "profile_phase_timer_names", ()))
        for key, value in timing_seconds.items():
            prefix = "timing/sampled" if key in sampled_phase_timers and self.profile_record_samples else "timing"
            self.logger.add_scalar(f"{prefix}/{key}_seconds/epoch", value, epoch)
        self._log_profile_metrics(epoch)
        if epoch == self.start_epoch:
            self.logger.add_scalar(
                "startup/dataset_load_seconds_local/epoch",
                self.dataset_load_seconds_local,
                epoch,
            )
            self.logger.add_scalar(
                "startup/dataset_load_seconds_mean/epoch",
                self.dataset_load_seconds_mean,
                epoch,
            )
            self.logger.add_scalar(
                "startup/dataset_load_seconds_max/epoch",
                self.dataset_load_seconds_max,
                epoch,
            )
            self.logger.add_scalar(
                "startup/dataset_statistics_seconds_local/epoch",
                self.dataset_statistics_seconds_local,
                epoch,
            )
            self.logger.add_scalar(
                "startup/dataset_statistics_seconds_mean/epoch",
                self.dataset_statistics_seconds_mean,
                epoch,
            )
            self.logger.add_scalar(
                "startup/dataset_statistics_seconds_max/epoch",
                self.dataset_statistics_seconds_max,
                epoch,
            )
            for rank, value in enumerate(self.dataset_load_seconds_by_rank):
                self.logger.add_scalar(f"startup/dataset_load_seconds_rank_{rank}/epoch", value, epoch)
            for rank, value in enumerate(self.dataset_statistics_seconds_by_rank):
                self.logger.add_scalar(
                    f"startup/dataset_statistics_seconds_rank_{rank}/epoch",
                    value,
                    epoch,
                )
        for valid_dataset_name, valid_loss in avg_valid_losses.items():
            self.logger.add_scalar(f"training/valid_{valid_dataset_name}_loss/epoch", valid_loss, epoch)
        for key, value in avg_train_loss_itemized.items():
            self.logger.add_scalar(f"training_info/{key}/epoch", value, epoch)
        for valid_dataset_name, valid_loss_itemized in avg_valid_losses_itemized.items():
            for key, value in valid_loss_itemized.items():
                self.logger.add_scalar(f"validating_info/{key}_{valid_dataset_name}/epoch", value, epoch)

    @torch.no_grad()
    def eval(self, epoch: int) -> None:
        if self.evaluator is None:
            return

        self.neural_model_unwrapped.eval()
        print_info("-" * 100)
        print_info("Evaluating")
        if self.action_mode == "action":
            _, _, error_stats = self.evaluator.evaluate_action_mode(
                num_traj=self.num_eval_rollouts,
                eval_mode="rollout",
                trajectory_source=self.eval_mode,
                zero_actions=self.eval_passive,
                render=self.eval_render,
            )
        elif self.action_mode == "joint_f":
            _, _, error_stats = self.evaluator.evaluate_joint_f_mode(
                num_traj=self.num_eval_rollouts,
                eval_mode="rollout",
                trajectory_source=self.eval_mode,
                passive=self.eval_passive,
                render=self.eval_render,
            )
        else:
            raise NotImplementedError(f"Action mode {self.action_mode} not implemented.")

        for error_metric_name, value in error_stats["overall"].items():
            if "std" not in error_metric_name:
                self.logger.add_scalar(f"eval_{self.eval_horizon}-steps/{error_metric_name}/epoch", value, epoch)
        for error_metric_name, values in error_stats["step-wise"].items():
            if "std" in error_metric_name:
                continue
            num_steps = values.shape[0]
            steps_to_log = range(num_steps) if num_steps <= 10 else np.linspace(0, num_steps - 1, 10, dtype=int)
            for step_index in steps_to_log:
                self.logger.add_scalar(
                    f"eval_details/{error_metric_name}_step_{step_index}/epoch",
                    values[step_index],
                    epoch,
                )

        eval_error = error_stats["overall"]["error(MSE)"]
        eval_error_value = float(eval_error.detach().cpu())
        print_info(
            "[Evaluate], Num Rollouts = {}, Rollout Length = {}, Rollout MSE Error = {:.8f}, "
            "Rollout MSE Error (joint_q) = {:.8f}".format(
                self.num_eval_rollouts,
                self.eval_horizon,
                eval_error_value,
                error_stats["overall"]["q_error(MSE)"],
            )
        )

        if eval_error_value < self.best_eval_error:
            self.best_eval_error = eval_error_value
            self.save_model("best_eval_model")
            with open(os.path.join(self.model_log_dir, "saved_best_eval_model_epochs.txt"), "a") as fp:
                fp.write(f"{epoch}\n")
            print_ok(f"Save Best Eval Model at Epoch {epoch} with MSE error {eval_error_value}.")

    def test(self) -> None:
        raise NotImplementedError("Dataset validation/test entry is not migrated yet; run train with --test disabled.")

    def save_model(self, filename: str | None = None) -> None:
        filename = filename or "best_model"
        training_state = {}
        if hasattr(self, "optimizer"):
            training_state["optimizer_state_dict"] = self.optimizer.state_dict()
        if hasattr(self, "current_epoch"):
            training_state["epoch"] = self.current_epoch
        if hasattr(self, "best_eval_error"):
            training_state["best_eval_error"] = self.best_eval_error
        if hasattr(self, "best_valid_losses"):
            training_state["best_valid_losses"] = self.best_valid_losses
        if hasattr(self, "loss_weights") and isinstance(self.loss_weights, torch.Tensor):
            training_state["loss_weights"] = self.loss_weights
        if hasattr(self, "contact_rms_counts"):
            training_state["contact_rms_counts"] = {
                key: value.detach().cpu() for key, value in self.contact_rms_counts.items()
            }
        checkpoint_path = os.path.join(self.model_log_dir, f"{filename}.pt")
        save_checkpoint(
            path=checkpoint_path,
            model=self.neural_model_unwrapped,
            robot_name=self.neural_env.robot_name,
            cfg=self.cfg,
            **training_state,
        )
        if self.is_main_process and self.logger.wandb and filename and "best" in filename:
            self.logger.log_checkpoint(checkpoint_path)


class SequenceModelTrainer(VanillaTrainer):
    """Sequence trainer using fixed-length trajectory windows."""

    def __init__(self, neural_env, cfg, checkpoint=None, device="cuda:0"):
        self.sample_sequence_length = cfg["algorithm"].get("sample_sequence_length", 1)
        super().__init__(neural_env, cfg, checkpoint, device)

    def get_datasets(self, train_dataset_path, valid_datasets_cfg) -> None:
        self.train_dataset_rank_sharded = self.is_distributed and self.dataset_load_mode == "eager"
        self.train_dataset = create_trajectory_dataset(
            load_mode=self.dataset_load_mode,
            sample_sequence_length=self.sample_sequence_length,
            hdf5_dataset_path=train_dataset_path,
            max_capacity=self.dataset_max_capacity,
            rank=self.rank if self.train_dataset_rank_sharded else 0,
            world_size=self.world_size if self.train_dataset_rank_sharded else 1,
            expected_contact_representation=self.dataset_contact_representation,
            require_solver_active=self.require_solver_active,
        )
        if valid_datasets_cfg is not None and (not self.is_distributed or self.is_main_process):
            for valid_dataset_name, valid_dataset_path in valid_datasets_cfg.items():
                self.valid_datasets[valid_dataset_name] = create_trajectory_dataset(
                    load_mode=self.dataset_load_mode,
                    sample_sequence_length=self.sample_sequence_length,
                    hdf5_dataset_path=valid_dataset_path,
                    expected_contact_representation=self.dataset_contact_representation,
                    require_solver_active=self.require_solver_active,
                )
        self.collate_fn = None
