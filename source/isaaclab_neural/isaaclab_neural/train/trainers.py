# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trainer implementations for NeRD neural dynamics models."""

from __future__ import annotations

import inspect
import math
import os
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import yaml
from newton import JointType
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

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
from isaaclab_neural.utils.checkpoint import reconstruct_model_from_checkpoint, save_checkpoint
from isaaclab_neural.utils.logger import Logger
from isaaclab_neural.utils.python_utils import format_dict, print_info, print_ok, print_warning, set_random_seed
from isaaclab_neural.utils.running_mean_std import RunningMeanStd
from isaaclab_neural.utils.time_report import TimeProfiler, TimeReport
from isaaclab_neural.utils.torch_utils import grad_norm, num_params_torch_model


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

        self.batch_size = int(algo_cfg["batch_size"])
        self.num_valid_batches = int(algo_cfg.get("num_valid_batches", 50))
        self.student_forcing_enabled = False
        dataset_cfg = algo_cfg["dataset"]
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
        self.get_datasets(algo_cfg["dataset"].get("train_dataset_path"), algo_cfg["dataset"].get("valid_datasets"))

        if cli_cfg["train"]:
            self.num_epochs = int(algo_cfg["num_epochs"])
            self.num_iters_per_epoch = int(algo_cfg.get("num_iters_per_epoch", -1))
            self.start_epoch = 0
            self.best_valid_losses = {}
            self.best_eval_error = np.inf

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

            self._wrap_distributed_model()
            self._init_optimizer(algo_cfg)
            if self._checkpoint is not None and self._checkpoint.get("version", 1) >= 2:
                self._restore_training_state()

            self._checkpoint = None
            self.truncate_grad = algo_cfg.get("truncate_grad", False)
            self.grad_norm = algo_cfg.get("grad_norm", 1.0)
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
        return {
            "env_name": env_cfg.get("env_name"),
            "robot_name": env_cfg.get("robot_name"),
            "num_envs": env_cfg.get("num_envs"),
            "contact_mode": solver_cfg.get("contact_mode"),
            "num_contacts_per_env": solver_cfg.get("num_contacts_per_env"),
            "algorithm": algo_cfg.get("name"),
            "batch_size": algo_cfg.get("batch_size"),
            "num_epochs": algo_cfg.get("num_epochs"),
            "lr_start": algo_cfg.get("optimizer", {}).get("lr_start"),
            "lr_end": algo_cfg.get("optimizer", {}).get("lr_end"),
            "lr_schedule": algo_cfg.get("optimizer", {}).get("lr_schedule"),
        }

    def _init_evaluator(self, algo_cfg: dict[str, Any], cli_cfg: dict[str, Any]) -> None:
        eval_cfg = algo_cfg.get("eval", {})
        cli_eval_interval = cli_cfg.get("eval_interval")
        self.eval_interval = eval_cfg.get("interval", 0) if cli_eval_interval is None else cli_eval_interval
        self.evaluator = None
        if self.eval_interval <= 0:
            return

        self.eval_mode = eval_cfg.get("mode", "dataset")
        self.action_mode = eval_cfg.get("action_mode", "joint_f")
        self.eval_horizon = int(eval_cfg.get("rollout_horizon", 5))
        self.num_eval_rollouts = int(eval_cfg.get("num_rollouts", self.neural_env.num_envs))
        self.eval_dataset_path = eval_cfg.get("dataset_path")
        self.eval_passive = bool(eval_cfg.get("passive", True))
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
            device=self.device,
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
                        contact_token_moments = ContactTokenMoments(value.shape[-1], self.device)
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

    def get_student_forcing_probability(self, epoch: int) -> float:
        """Return the student forcing probability for trainers that support it."""
        raise NotImplementedError("Student forcing is not supported by this trainer.")

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
    def preprocess_data_batch(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key, value in data.items():
            data[key] = value.to(self.device, non_blocking=self.non_blocking_data_transfer)
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

    def compute_loss(self, data: dict[str, torch.Tensor], train: bool):
        if self.neural_model_unwrapped.is_rnn:
            self.neural_model_unwrapped.init_rnn(self.batch_size)
        prediction_target = data["prediction_target"]
        prediction = self._forward_model(data, train)
        loss = self.loss_func(prediction * self.loss_weights, prediction_target * self.loss_weights)

        with torch.no_grad():
            predicted_next_states = self.neural_solver.convert_prediction_to_next_states(
                states=data["states"],
                prediction=prediction,
                dt=self.neural_env.frame_dt,
            )
            self.neural_solver.wrap2PI(predicted_next_states)
            loss_itemized = {
                "state_MSE": torch.nn.MSELoss()(predicted_next_states, data["next_states"]).detach().cpu().item(),
                "q_error_norm": torch.norm(
                    predicted_next_states[..., : self.neural_solver.dof_q_per_env]
                    - data["next_states"][..., : self.neural_solver.dof_q_per_env],
                    dim=-1,
                )
                .mean()
                .detach()
                .cpu()
                .item(),
                "qd_error_norm": torch.norm(
                    predicted_next_states[..., self.neural_solver.dof_q_per_env :]
                    - data["next_states"][..., self.neural_solver.dof_q_per_env :],
                    dim=-1,
                )
                .mean()
                .detach()
                .cpu()
                .item(),
            }
        return loss, loss_itemized

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
        sum_loss = torch.tensor(0.0, device=self.device)
        sum_loss_itemized: dict[str, float] = {}
        grad_info = {"grad_norm_before_clip": 0.0} if train else {}
        if train and self.truncate_grad:
            grad_info["grad_norm_after_clip"] = 0.0

        with torch.set_grad_enabled(train):
            for _ in tqdm(range(num_batches), disable=not self.is_main_process):
                with TimeProfiler(self.time_report, "dataloader"):
                    try:
                        data = next(dataloader_iter)
                    except StopIteration:
                        if shuffle and self.train_dataset is not None:
                            self.train_dataset.shuffle()
                        dataloader_iter = iter(dataloader)
                        data = next(dataloader_iter)
                    data = self.preprocess_data_batch(data)

                with TimeProfiler(self.time_report, "compute_loss"):
                    if train:
                        self.optimizer.zero_grad()
                    loss, loss_itemized = self.compute_loss(data, train)

                with TimeProfiler(self.time_report, "backward"):
                    if train:
                        loss.backward()
                        with torch.no_grad():
                            grad_norm_before_clip = float(grad_norm(self.neural_model.parameters()).detach().cpu())
                            grad_info["grad_norm_before_clip"] += grad_norm_before_clip
                            if self.truncate_grad:
                                clip_grad_norm_(self.neural_model.parameters(), self.grad_norm)
                                grad_info["grad_norm_after_clip"] += float(
                                    grad_norm(self.neural_model.parameters()).detach().cpu()
                                )
                        self.optimizer.step()

                with TimeProfiler(self.time_report, "other"):
                    sum_loss += loss.detach()
                    for key, value in loss_itemized.items():
                        sum_loss_itemized[key] = sum_loss_itemized.get(key, 0.0) + value

        avg_loss = sum_loss.cpu().item() / num_batches
        avg_loss_itemized = {key: value / num_batches for key, value in sum_loss_itemized.items()}
        if train:
            for key in grad_info:
                grad_info[key] /= num_batches
        if distributed_reduce:
            avg_loss = self._distributed_mean(avg_loss)
            avg_loss_itemized = self._distributed_mean_dict(avg_loss_itemized)
            grad_info = self._distributed_mean_dict(grad_info)
        return avg_loss, avg_loss_itemized, grad_info

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

        self.time_report = TimeReport(cuda_synchronize=False)
        self.time_report.add_timers(["epoch", "other", "dataloader", "compute_loss", "backward", "eval"])
        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loader_iter = iter(train_loader)
            self.time_report.reset_timer()
            with TimeProfiler(self.time_report, "epoch"):
                self.lr = self.get_scheduled_learning_rate(epoch, self.num_epochs)
                if self.student_forcing_enabled:
                    self.p_student = self.get_student_forcing_probability(epoch)
                    print_info(f"Student forcing probability: {self.p_student}")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
                self.logger.init_epoch(epoch)

                avg_train_loss, avg_train_loss_itemized, grad_info = self.one_epoch(
                    train=True,
                    dataloader=train_loader,
                    dataloader_iter=train_loader_iter,
                    num_batches=self.num_train_batches,
                    shuffle=True,
                )
                avg_valid_losses, avg_valid_losses_itemized = {}, {}
                if self.is_main_process:
                    valid_loader_iters = _reset_validation_iterators(valid_loaders)
                    for valid_dataset_name in self.valid_datasets:
                        avg_valid_losses[valid_dataset_name], avg_valid_losses_itemized[valid_dataset_name], _ = (
                            self.one_epoch(
                                train=False,
                                dataloader=valid_loaders[valid_dataset_name],
                                dataloader_iter=valid_loader_iters[valid_dataset_name],
                                num_batches=min(self.num_valid_batches, len(valid_loaders[valid_dataset_name])),
                                distributed_reduce=False,
                            )
                        )
                with TimeProfiler(self.time_report, "eval"):
                    if self.is_main_process and self.eval_interval > 0 and (epoch + 1) % self.eval_interval == 0:
                        self.eval(epoch)
                    if self.is_distributed:
                        dist.barrier()

            if self.is_main_process and epoch % self.log_interval == 0:
                self._log_training_epoch_summary(
                    epoch,
                    avg_train_loss,
                    avg_train_loss_itemized,
                    avg_valid_losses,
                    avg_valid_losses_itemized,
                    grad_info,
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

        if self.is_main_process:
            self.save_model("final_model")
            self.logger.finish()

    def _log_training_epoch_summary(
        self,
        epoch: int,
        avg_train_loss: float,
        avg_train_loss_itemized: dict[str, float],
        avg_valid_losses: dict[str, float],
        avg_valid_losses_itemized: dict[str, dict[str, float]],
        grad_info: dict[str, float],
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
        self.logger.add_scalar("params/lr/epoch", self.lr, epoch)
        self.logger.add_scalar("training/train_loss/epoch", avg_train_loss, epoch)
        if "grad_norm_before_clip" in grad_info:
            self.logger.add_scalar("training/gradients_before_clip/epoch", grad_info["grad_norm_before_clip"], epoch)
        if "grad_norm_after_clip" in grad_info:
            self.logger.add_scalar("training/gradients_after_clip/epoch", grad_info["grad_norm_after_clip"], epoch)
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
        )
        if valid_datasets_cfg is not None and (not self.is_distributed or self.is_main_process):
            for valid_dataset_name, valid_dataset_path in valid_datasets_cfg.items():
                self.valid_datasets[valid_dataset_name] = create_trajectory_dataset(
                    load_mode=self.dataset_load_mode,
                    sample_sequence_length=self.sample_sequence_length,
                    hdf5_dataset_path=valid_dataset_path,
                )
        self.collate_fn = None


class MultiStepTrainer(SequenceModelTrainer):
    """Multi-step trainer with scheduled student forcing in world frame."""

    def __init__(self, neural_env, cfg, checkpoint=None, device="cuda:0"):
        super().__init__(neural_env, cfg, checkpoint, device)
        if cfg["cli"]["train"]:
            student_forcing_cfg = cfg["algorithm"].get("student_forcing", {})
            self.student_forcing_enabled = True
            self.student_forcing_schedule = student_forcing_cfg.get("schedule", "linear")
            self.student_forcing_start_epoch = student_forcing_cfg.get("start_epoch", 0)
            self.student_forcing_end_epoch = student_forcing_cfg.get("end_epoch", self.num_epochs)
            self.student_forcing_start_prob = student_forcing_cfg.get("start_prob", 0.0)
            self.student_forcing_end_prob = student_forcing_cfg.get("end_prob", 1.0)

    @torch.no_grad()
    def preprocess_data_batch(self, data):
        for key, value in data.items():
            data[key] = value.to(self.device, non_blocking=self.non_blocking_data_transfer)
        data["states_w"] = data["states"].clone()
        data["next_states_w"] = data["next_states"].clone()
        data["gravity_dir_w"] = data["gravity_dir"].clone()
        return super().preprocess_data_batch(data)

    def get_student_forcing_probability(self, epoch: int) -> float:
        if epoch < self.student_forcing_start_epoch:
            return self.student_forcing_start_prob
        if epoch >= self.student_forcing_end_epoch:
            return self.student_forcing_end_prob
        progress = (epoch - self.student_forcing_start_epoch) / (
            self.student_forcing_end_epoch - self.student_forcing_start_epoch
        )
        if self.student_forcing_schedule == "linear":
            return self.student_forcing_start_prob + progress * (
                self.student_forcing_end_prob - self.student_forcing_start_prob
            )
        if self.student_forcing_schedule == "exponential":
            return self.student_forcing_start_prob + progress**2 * (
                self.student_forcing_end_prob - self.student_forcing_start_prob
            )
        if self.student_forcing_schedule == "sigmoid":
            sigmoid = 1 / (1 + math.exp(-(progress - 0.5) * 12))
            return self.student_forcing_start_prob + sigmoid * (
                self.student_forcing_end_prob - self.student_forcing_start_prob
            )
        raise ValueError(f"Unknown schedule: {self.student_forcing_schedule}")

    def compute_loss(self, data, train):
        batch_size, seq_length, _ = data["states_w"].shape
        device = data["states_w"].device
        if not hasattr(self, "p_student"):
            self.p_student = self.get_student_forcing_probability(self.current_epoch)
        if self.neural_model_unwrapped.is_rnn:
            self.neural_model_unwrapped.init_rnn(batch_size)

        total_loss = 0.0
        input_states_w = data["states_w"][:, 0:1, :].clone()
        all_predicted_next_states_w = []
        for time_index in range(seq_length):
            if self.neural_solver.base_joint_type == JointType.FREE:
                root_body_q = torch.cat([input_states_w[..., :3], input_states_w[..., 3:7]], dim=-1)
            else:
                root_body_q = torch.zeros((batch_size, time_index + 1, 7), device=device)
            input_states_model, _, _, _, _, input_gravity_dir_model = self.neural_solver.convert_coordinate_frame(
                root_body_q=root_body_q,
                states=input_states_w,
                next_states=None,
                contact_points_0=None,
                contact_points_1=None,
                contact_normals=None,
                gravity_dir=data["gravity_dir_w"][:, : time_index + 1, :],
            )
            model_inputs = {
                "states": input_states_model,
                "states_embedding": self.neural_solver.embed_states(input_states_model),
                "gravity_dir": input_gravity_dir_model,
                "root_body_q": root_body_q,
            }
            exclude_keys = {
                "states",
                "states_embedding",
                "gravity_dir",
                "root_body_q",
                "states_w",
                "next_states_w",
                "gravity_dir_w",
            }
            for key, value in data.items():
                if key not in exclude_keys:
                    model_inputs[key] = value[:, : time_index + 1, :]
            prediction = self._forward_model(model_inputs, train, single_step=True)
            predicted_next_state_model = self.neural_solver.convert_prediction_to_next_states(
                states=input_states_model[:, -1, :],
                prediction=prediction.squeeze(1),
                dt=self.neural_env.frame_dt,
            )
            predicted_next_state_w = self.neural_solver.convert_states_back_to_world(
                root_body_q=root_body_q,
                states=predicted_next_state_model,
            )
            predicted_next_state_w = self.neural_solver.wrap2PI_differentiable(predicted_next_state_w)
            total_loss += self.loss_func(
                predicted_next_state_w * self.loss_weights,
                data["next_states_w"][:, time_index, :] * self.loss_weights,
            )
            all_predicted_next_states_w.append(predicted_next_state_w.detach())
            if time_index < seq_length - 1:
                use_student = torch.rand(batch_size, device=device) < self.p_student
                next_state_w = torch.where(
                    use_student.view(-1, 1), predicted_next_state_w, data["states_w"][:, time_index + 1, :]
                )
                input_states_w = torch.cat([input_states_w, next_state_w.unsqueeze(1)], dim=1)

        loss = total_loss / seq_length
        with torch.no_grad():
            predicted = torch.stack(all_predicted_next_states_w, dim=1)
            loss_itemized = self._sequence_loss_itemized(predicted, data["next_states_w"])
            loss_itemized["student_forcing_prob"] = self.p_student
        return loss, loss_itemized

    def _sequence_loss_itemized(self, predicted_next_states, next_states):
        return {
            "state_MSE": torch.nn.MSELoss()(predicted_next_states, next_states).detach().cpu().item(),
            "q_error_norm": torch.norm(
                predicted_next_states[..., : self.neural_solver.dof_q_per_env]
                - next_states[..., : self.neural_solver.dof_q_per_env],
                dim=-1,
            )
            .mean()
            .detach()
            .cpu()
            .item(),
            "qd_error_norm": torch.norm(
                predicted_next_states[..., self.neural_solver.dof_q_per_env :]
                - next_states[..., self.neural_solver.dof_q_per_env :],
                dim=-1,
            )
            .mean()
            .detach()
            .cpu()
            .item(),
        }


class MultiStepTrainerNew(MultiStepTrainer):
    """Multi-step trainer variant for first-anchor-frame configs."""

    def __init__(self, neural_env, cfg, checkpoint=None, device="cuda:0"):
        if cfg["env"]["neural_solver_cfg"].get("anchor_frame_step") != "first":
            raise ValueError("anchor_frame_step must be 'first' for MultiStepTrainerNew")
        super().__init__(neural_env, cfg, checkpoint, device)

    def compute_loss(self, data, train):
        batch_size, seq_length, _ = data["states"].shape
        device = data["states"].device
        if not hasattr(self, "p_student"):
            self.p_student = self.get_student_forcing_probability(self.current_epoch)
        if self.p_student < 1e-6:
            return VanillaTrainer.compute_loss(self, data, train)
        if self.neural_model_unwrapped.is_rnn:
            self.neural_model_unwrapped.init_rnn(batch_size)

        total_loss = 0.0
        input_states = data["states"][:, 0:1, :].clone()
        all_predicted_next_states = []
        for time_index in range(seq_length):
            input_states_embedding = self.neural_solver.embed_states(input_states)
            model_inputs = {
                "states": input_states,
                "states_embedding": input_states_embedding,
            }
            for key, value in data.items():
                if key not in {"states", "states_embedding"}:
                    model_inputs[key] = value[:, : time_index + 1, :]

            prediction = self._forward_model(model_inputs, train, single_step=True)
            predicted_next_states = self.neural_solver.convert_prediction_to_next_states(
                states=input_states[:, -1, :],
                prediction=prediction.squeeze(1),
                dt=self.neural_env.frame_dt,
            )
            predicted_next_states = self.neural_solver.wrap2PI_differentiable(predicted_next_states)
            total_loss += self.loss_func(
                predicted_next_states * self.loss_weights,
                data["next_states"][:, time_index, :] * self.loss_weights,
            )
            all_predicted_next_states.append(predicted_next_states.detach())

            if time_index < seq_length - 1:
                use_student = torch.rand(batch_size, device=device) < self.p_student
                next_state = torch.where(
                    use_student.view(-1, 1),
                    predicted_next_states,
                    data["states"][:, time_index + 1, :],
                )
                input_states = torch.cat([input_states, next_state.unsqueeze(1)], dim=1)

        loss = total_loss / seq_length
        with torch.no_grad():
            predicted = torch.stack(all_predicted_next_states, dim=1)
            loss_itemized = self._sequence_loss_itemized(predicted, data["next_states"])
            loss_itemized["student_forcing_prob"] = self.p_student
        return loss, loss_itemized
