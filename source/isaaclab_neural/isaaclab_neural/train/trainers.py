# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trainer implementations for NeRD neural dynamics models."""

from __future__ import annotations

import inspect
import os
import shutil
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

from isaaclab_neural.contacts.tensor_utils import (
    MIN_CONTACT_RMS_SAMPLES,
    ContactTokenMoments,
    MaskedContactMoments,
)
from isaaclab_neural.data import (
    collate_fn_BatchTransitionDataset,
    create_batch_transition_dataset,
    create_trajectory_dataset,
    create_weighted_trajectory_dataset,
)
from isaaclab_neural.eval.training_evaluator import TrainingRolloutEvaluator
from isaaclab_neural.models.models import ModelMixedInput
from isaaclab_neural.utils import torch_utils
from isaaclab_neural.utils.checkpoint import reconstruct_model_from_checkpoint, save_checkpoint
from isaaclab_neural.utils.logger import Logger
from isaaclab_neural.utils.python_utils import format_dict, print_info, print_ok, print_warning, set_random_seed
from isaaclab_neural.utils.running_mean_std import RunningMeanStd
from isaaclab_neural.utils.time_report import TimeProfiler, TimeReport
from isaaclab_neural.utils.torch_utils import grad_norm, num_params_torch_model

BODY_FORCE_LABEL_KEY = "body_net_contact_forces_w"
BODY_FORCE_TARGET_KEY = "body_net_contact_forces_b"
# A body is considered in active contact when its target net-force magnitude
# exceeds this value [N]. Recipes may override it under algorithm.loss.
DEFAULT_ACTIVE_CONTACT_FORCE_THRESHOLD_N = 1.0

NON_MODEL_DATA_KEYS = {
    "physics_step",
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


def transform_body_net_contact_forces_to_current_root_frame(
    forces_w: torch.Tensor,
    root_body_q: torch.Tensor,
) -> torch.Tensor:
    """Rotate world-frame per-body resultant forces into the current root-body axes.

    Args:
        forces_w: Resultant linear contact forces [N], including normal and
            tangential/friction components but excluding contact moments and
            torques. Shape is ``[B, T, num_bodies, 3]``.
        root_body_q: Current root poses, shape ``[B, T, 7]``, with quaternion
            convention matching the solver's body-frame preprocessing.

    Returns:
        Forces in current root-body axes [N], shape ``[B, T, num_bodies, 3]``.
    """
    if forces_w.ndim != 4 or forces_w.shape[-1] != 3:
        raise ValueError(f"{BODY_FORCE_LABEL_KEY} must have shape [B, T, num_bodies, 3], got {tuple(forces_w.shape)}.")
    if root_body_q.shape != (*forces_w.shape[:2], 7):
        raise ValueError(
            f"root_body_q must have shape [B, T, 7] matching force labels, got {tuple(root_body_q.shape)}."
        )
    body_frame_quat = root_body_q[..., 3:7].unsqueeze(-2).expand(*forces_w.shape[:-1], 4)
    return torch_utils.quat_rotate_inverse(
        body_frame_quat.reshape(-1, 4),
        forces_w.reshape(-1, 3),
    ).view_as(forces_w)


@torch.no_grad()
def body_net_contact_force_metrics(
    prediction_b: torch.Tensor,
    target_b: torch.Tensor,
    active_contact_threshold_n: float = DEFAULT_ACTIVE_CONTACT_FORCE_THRESHOLD_N,
) -> dict[str, float]:
    """Compute physical-unit metrics for per-body resultant-force predictions.

    Inputs are linear contact forces [N] containing normal and
    tangential/friction components, with no contact moments or torques.
    """
    error_norm_sq = (prediction_b - target_b).square().sum(dim=-1)
    per_body_rmse = error_norm_sq.mean(dim=(0, 1)).sqrt()
    active_mask = target_b.norm(dim=-1) > active_contact_threshold_n
    active_rmse = error_norm_sq[active_mask].mean().sqrt() if active_mask.any() else error_norm_sq.new_zeros(())
    support_error = (prediction_b.sum(dim=-2) - target_b.sum(dim=-2)).square().sum(dim=-1)
    all_body_rmse = error_norm_sq.mean().sqrt()
    metrics = {
        "body_force_RMSE_N": float(all_body_rmse.cpu()),
        # Rotation preserves vector error norms, so this is also the error after
        # reconstructing the ContactSensor world-frame output.
        "body_force_contact_sensor_reconstruction_RMSE_N": float(all_body_rmse.cpu()),
        "body_force_active_RMSE_N": float(active_rmse.cpu()),
        "body_force_per_body_RMSE_mean_N": float(per_body_rmse.mean().cpu()),
        "body_force_per_body_RMSE_max_N": float(per_body_rmse.max().cpu()),
        "body_force_support_consistency_RMSE_N": float(support_error.mean().sqrt().cpu()),
        "body_force_active_fraction": float(active_mask.float().mean().cpu()),
    }
    metrics.update(
        {
            f"body_force_RMSE_body_{body_index:02d}_N": float(value.cpu())
            for body_index, value in enumerate(per_body_rmse)
        }
    )
    return metrics


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

        force_head_cfg = cfg["network"].get("body_net_contact_force_head", {})
        self.force_head_enabled = bool(getattr(self.neural_model, "predict_body_net_contact_forces", False))
        self.finger_force_head_enabled = bool(getattr(self.neural_model, "predict_finger_object_normal_forces", False))
        requested_finger_head = cfg["network"].get("finger_object_normal_force_head", {}).get("enabled", False)
        if requested_finger_head != self.finger_force_head_enabled:
            raise ValueError("The checkpoint's filtered finger-force head does not match the training configuration.")
        self.finger_force_loss_weight = float(
            algo_cfg.get("loss", {}).get("finger_object_normal_force", {}).get("weight", 1.0)
        )
        force_loss_cfg = algo_cfg.get("loss", {}).get("body_net_contact_force", {})
        self.force_loss_weight = float(force_loss_cfg.get("weight", force_head_cfg.get("loss_weight", 1.0)))
        self.active_contact_force_threshold_n = float(
            force_loss_cfg.get(
                "active_contact_threshold_n",
                force_head_cfg.get(
                    "active_contact_threshold_n",
                    DEFAULT_ACTIVE_CONTACT_FORCE_THRESHOLD_N,
                ),
            )
        )
        if self.force_loss_weight < 0.0:
            raise ValueError("Body net contact-force loss weight must be non-negative.")
        if self.active_contact_force_threshold_n < 0.0:
            raise ValueError("Active-contact force threshold [N] must be non-negative.")

        self.batch_size = int(algo_cfg["batch_size"])
        self.num_valid_batches = int(algo_cfg.get("num_valid_batches", 50))
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
        solver_cfg = cfg["env"]["neural_solver_cfg"]
        self.dataset_contact_projection = {
            "contact_representation": getattr(self.neural_solver, "contact_representation", None),
            "num_contacts_per_env": int(getattr(self.neural_solver, "num_contacts_per_env", 0)),
            "max_contact_tokens": int(getattr(self.neural_solver, "max_contact_tokens", 0)),
            "contact_packing_policy": solver_cfg.get("contact_packing_policy"),
            "contact_filter": solver_cfg.get("contact_filter", "none"),
        }
        dataset_cfg = algo_cfg["dataset"]
        self.get_datasets(
            dataset_cfg.get("train_dataset_path"),
            dataset_cfg.get("valid_datasets"),
            dataset_cfg.get("train_datasets"),
        )
        self._validate_force_label_datasets()

        if cli_cfg["train"]:
            self.num_epochs = int(algo_cfg["num_epochs"])
            self.num_iters_per_epoch = int(algo_cfg.get("num_iters_per_epoch", -1))
            self.start_epoch = 0
            self.best_valid_losses = {}
            self.best_eval_error = np.inf

            force_rms_missing = (
                self.force_head_enabled and getattr(self.neural_model, "body_net_contact_force_rms", None) is None
            ) or (
                self.finger_force_head_enabled
                and getattr(self.neural_model, "finger_object_normal_force_rms", None) is None
            )
            checkpoint_loss_weights_missing = self._checkpoint is not None and "loss_weights" not in self._checkpoint
            if (
                algo_cfg.get("update_dataset_statistics", True)
                or self._checkpoint is None
                or getattr(self.neural_model, "input_rms", None) is None
                or getattr(self.neural_model, "output_rms", None) is None
                or force_rms_missing
                or checkpoint_loss_weights_missing
            ):
                self.compute_or_sync_dataset_statistics(self.train_dataset)
                self.neural_model_unwrapped.set_input_rms(self.dataset_rms)
                self.neural_model_unwrapped.set_output_rms(self.dataset_rms["prediction_target"])
                if self.force_head_enabled:
                    self.neural_model_unwrapped.set_body_net_contact_force_rms(self.dataset_rms[BODY_FORCE_TARGET_KEY])
                if self.finger_force_head_enabled:
                    self.neural_model_unwrapped.set_finger_object_normal_force_rms(
                        self.dataset_rms["finger_object_normal_forces_b"]
                    )
                self.loss_weights = (
                    1.0 / torch.sqrt(self.dataset_rms["relative_states"].var + 1e-5)
                    if algo_cfg.get("weighted_loss", True)
                    else 1.0
                )
            else:
                print_info("Using dataset statistics from checkpoint...")
                self.loss_weights = self._checkpoint["loss_weights"].to(self.device)

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
        inputs_cfg = self.cfg.get("inputs", {})
        contact_cfg = inputs_cfg.get("contact_set", {})
        network_cfg = self.cfg.get("network", {})
        transformer_cfg = network_cfg.get("transformer", {})
        model_cfg = network_cfg.get("model", {})
        force_head_cfg = network_cfg.get("body_net_contact_force_head", {})
        force_loss_cfg = algo_cfg.get("loss", {}).get("body_net_contact_force", {})
        return {
            "env_name": env_cfg.get("env_name"),
            "robot_name": env_cfg.get("robot_name"),
            "num_envs": env_cfg.get("num_envs"),
            "contact_mode": solver_cfg.get("contact_mode"),
            "num_contacts_per_env": solver_cfg.get("num_contacts_per_env"),
            "algorithm": algo_cfg.get("name"),
            "seed": self.seed,
            "batch_size": algo_cfg.get("batch_size"),
            "num_epochs": algo_cfg.get("num_epochs"),
            "num_iters_per_epoch": getattr(self, "num_iters_per_epoch", algo_cfg.get("num_iters_per_epoch")),
            "dataset_max_capacity": self.dataset_max_capacity,
            "lr_start": algo_cfg.get("optimizer", {}).get("lr_start"),
            "lr_end": algo_cfg.get("optimizer", {}).get("lr_end"),
            "lr_schedule": algo_cfg.get("optimizer", {}).get("lr_schedule"),
            "model_num_parameters": num_params_torch_model(self.neural_model),
            "contact_encoder_type": contact_cfg.get("encoder_type"),
            "contact_body_latent_dim": contact_cfg.get("body_latent_dim"),
            "contact_hidden_dim": contact_cfg.get("hidden_dim"),
            "transformer_n_layer": transformer_cfg.get("n_layer"),
            "transformer_n_head": transformer_cfg.get("n_head"),
            "transformer_n_embd": transformer_cfg.get("n_embd"),
            "model_mlp_layer_sizes": model_cfg.get("mlp", {}).get("layer_sizes"),
            "body_net_contact_force_head_enabled": getattr(
                self,
                "force_head_enabled",
                bool(force_head_cfg.get("enabled", False)),
            ),
            "body_net_contact_force_loss_weight": getattr(
                self,
                "force_loss_weight",
                float(force_loss_cfg.get("weight", force_head_cfg.get("loss_weight", 1.0))),
            ),
            "body_net_contact_force_active_threshold_n": getattr(
                self,
                "active_contact_force_threshold_n",
                float(
                    force_loss_cfg.get(
                        "active_contact_threshold_n",
                        force_head_cfg.get(
                            "active_contact_threshold_n",
                            DEFAULT_ACTIVE_CONTACT_FORCE_THRESHOLD_N,
                        ),
                    )
                ),
            ),
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
            device=self.device,
            require_terrain_context=self.eval_require_terrain_context,
            rebase_env_origins=bool(eval_cfg.get("rebase_env_origins", False)),
            preserve_source_origins=bool(eval_cfg.get("preserve_source_origins", False)),
            contact_context_validation=self.eval_contact_context_validation,
            state_context_tolerance=self.eval_state_context_tolerance,
            contact_context_tolerance=self.eval_contact_context_tolerance,
            contact_context_normal_tolerance=self.eval_contact_context_normal_tolerance,
            contact_context_velocity_tolerance=self.eval_contact_context_velocity_tolerance,
            **self.dataset_contact_projection,
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

    def get_datasets(self, train_dataset_path, valid_datasets_cfg, train_datasets_cfg=None) -> None:
        if train_datasets_cfg is not None:
            raise ValueError("Weighted train_datasets are only supported by SequenceModelTrainer.")
        self.train_dataset = create_batch_transition_dataset(
            load_mode=self.dataset_load_mode,
            batch_size=self.batch_size,
            hdf5_dataset_path=train_dataset_path,
            max_capacity=self.dataset_max_capacity,
            device=self.device,
            **self.dataset_contact_projection,
        )
        if valid_datasets_cfg is not None and (not self.is_distributed or self.is_main_process):
            for valid_dataset_name, valid_dataset_path in valid_datasets_cfg.items():
                self.valid_datasets[valid_dataset_name] = create_batch_transition_dataset(
                    load_mode=self.dataset_load_mode,
                    batch_size=self.batch_size,
                    hdf5_dataset_path=valid_dataset_path,
                    device=self.device,
                    **self.dataset_contact_projection,
                )
        self.batch_size = 1
        self.collate_fn = collate_fn_BatchTransitionDataset

    def _validate_force_label_datasets(self) -> None:
        """Require force labels in every dataset when the auxiliary head is enabled."""
        if getattr(self, "finger_force_head_enabled", False):
            missing_finger = [
                name
                for name, dataset in {"train": self.train_dataset, **self.valid_datasets}.items()
                if dataset is not None and "finger_object_normal_forces_w" not in getattr(dataset, "data_keys", ())
            ]
            if missing_finger:
                raise ValueError(
                    f"Filtered finger-force head requires finger_object_normal_forces_w labels: {missing_finger}."
                )
        if not self.force_head_enabled:
            return
        datasets = {"train": self.train_dataset, **self.valid_datasets}
        missing = [
            name
            for name, dataset in datasets.items()
            if dataset is not None and BODY_FORCE_LABEL_KEY not in getattr(dataset, "data_keys", ())
        ]
        if missing:
            raise ValueError(
                f"Enabled body_net_contact_force_head requires {BODY_FORCE_LABEL_KEY!r} labels; "
                f"missing from datasets: {missing}."
            )
        body_name_orders = {
            name: tuple(getattr(dataset, "body_net_contact_force_body_names", ()) or ())
            for name, dataset in datasets.items()
            if dataset is not None
        }
        reference_name, reference_order = next(iter(body_name_orders.items()))
        mismatched_orders = {
            name: order for name, order in body_name_orders.items() if not order or order != reference_order
        }
        if mismatched_orders:
            raise ValueError(
                "Body net contact-force datasets must use the same non-empty body-name order; "
                f"{reference_name}={reference_order}, mismatched={mismatched_orders}."
            )

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
            kwargs["multiprocessing_context"] = "spawn"
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
        for key in NON_MODEL_DATA_KEYS.intersection(data):
            data.pop(key)
        if not self.force_head_enabled:
            data.pop(BODY_FORCE_LABEL_KEY, None)
        if not getattr(self, "finger_force_head_enabled", False):
            data.pop("finger_object_normal_forces_w", None)
        for key, value in data.items():
            data[key] = value.to(self.device, non_blocking=self.non_blocking_data_transfer)
        if "contact_masks" in data:
            data["contact_masks"] = data["contact_masks"].bool()
        elif "contact_tokens" not in data and "contact_depths" in data:
            raise ValueError(
                "Flat native-contact datasets require explicit contact_masks. "
                "Regenerate the dataset with the current generator."
            )
        if self.force_head_enabled:
            if BODY_FORCE_LABEL_KEY not in data:
                raise ValueError(f"Enabled body_net_contact_force_head requires {BODY_FORCE_LABEL_KEY!r} labels.")
            force_target = transform_body_net_contact_forces_to_current_root_frame(
                data.pop(BODY_FORCE_LABEL_KEY),
                data["root_body_q"],
            )
            expected_num_bodies = self.neural_model_unwrapped.num_bodies
            if force_target.shape[-2] != expected_num_bodies:
                raise ValueError(
                    f"{BODY_FORCE_LABEL_KEY} contains {force_target.shape[-2]} bodies, "
                    f"but the model force head expects {expected_num_bodies}."
                )
            data[BODY_FORCE_TARGET_KEY] = force_target
        if getattr(self, "finger_force_head_enabled", False):
            force_w = data.pop("finger_object_normal_forces_w", None)
            if force_w is None or force_w.shape[-2:] != (2, 3) or not torch.isfinite(force_w).all():
                raise ValueError("Filtered finger-force training requires finite [B, T, 2, 3] labels.")
            data["finger_object_normal_forces_b"] = transform_body_net_contact_forces_to_current_root_frame(
                force_w, data["root_body_q"]
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
        finger_enabled = getattr(self, "finger_force_head_enabled", False)
        model_output = self._forward_model(data, train, return_aux=self.force_head_enabled or finger_enabled)
        if self.force_head_enabled or finger_enabled:
            prediction = model_output.state_prediction
        else:
            prediction = model_output
        loss = self.loss_func(prediction * self.loss_weights, prediction_target * self.loss_weights)
        force_prediction = None
        force_target = None
        force_loss = None
        if self.force_head_enabled:
            force_prediction = model_output.auxiliary_predictions[BODY_FORCE_TARGET_KEY]
            force_target = data[BODY_FORCE_TARGET_KEY]
            force_rms = self.neural_model_unwrapped.body_net_contact_force_rms
            if force_rms is None:
                raise RuntimeError("Force-head training requires body net contact-force RMS statistics.")
            normalized_prediction = force_rms.normalize(force_prediction)
            normalized_target = force_rms.normalize(force_target)
            force_loss = torch.nn.functional.mse_loss(normalized_prediction, normalized_target)
            loss = loss + self.force_loss_weight * force_loss

        finger_prediction: torch.Tensor | None = None
        finger_target: torch.Tensor | None = None
        finger_loss: torch.Tensor | None = None
        contact_loss: torch.Tensor | None = None
        if finger_enabled:
            finger_prediction = model_output.auxiliary_predictions["finger_object_normal_forces_b"]
            finger_target = data["finger_object_normal_forces_b"]
            assert finger_target is not None
            finger_rms = self.neural_model_unwrapped.finger_object_normal_force_rms
            if finger_rms is None:
                raise RuntimeError("Filtered finger-force training requires its RMS statistics.")
            raw_force_prediction = model_output.auxiliary_predictions["finger_object_normal_forces_raw_b"]
            finger_loss = cast(
                torch.Tensor,
                torch.nn.functional.mse_loss(
                    finger_rms.normalize(raw_force_prediction), finger_rms.normalize(finger_target)
                ),
            )
            contact_loss = cast(
                torch.Tensor,
                torch.nn.functional.binary_cross_entropy_with_logits(
                    model_output.auxiliary_predictions["finger_object_contact_logits"],
                    (finger_target.norm(dim=-1) > 0.01).to(finger_target.dtype),
                ),
            )
            loss = loss + self.finger_force_loss_weight * (finger_loss + contact_loss)

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
            if force_prediction is not None and force_target is not None and force_loss is not None:
                loss_itemized["body_force_normalized_MSE"] = float(force_loss.detach().cpu())
                loss_itemized.update(
                    body_net_contact_force_metrics(
                        force_prediction,
                        force_target,
                        self.active_contact_force_threshold_n,
                    )
                )
            if (
                finger_prediction is not None
                and finger_target is not None
                and finger_loss is not None
                and contact_loss is not None
            ):
                error_sq = (finger_prediction - finger_target).square().sum(-1)
                active = finger_target.norm(dim=-1) > 0.01
                predicted_active = finger_prediction.norm(dim=-1) > 0.01
                true_positive = (active & predicted_active).sum()
                loss_itemized.update(
                    {
                        "finger_normal_force_normalized_MSE": float(finger_loss.detach().cpu()),
                        "finger_normal_contact_BCE": float(contact_loss.detach().cpu()),
                        "finger_normal_force_RMSE_N": float(error_sq.mean().sqrt().cpu()),
                        "finger_normal_force_active_RMSE_N": float(error_sq[active].mean().sqrt().cpu())
                        if active.any()
                        else 0.0,
                        "finger_normal_force_right_RMSE_N": float(error_sq[..., 0].mean().sqrt().cpu()),
                        "finger_normal_force_left_RMSE_N": float(error_sq[..., 1].mean().sqrt().cpu()),
                        "finger_normal_contact_precision": float(
                            (true_positive / predicted_active.sum().clamp_min(1)).cpu()
                        ),
                        "finger_normal_contact_recall": float((true_positive / active.sum().clamp_min(1)).cpu()),
                    }
                )
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
                main_process_exception = None
                main_process_traceback = None
                if self.is_main_process:
                    try:
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

    def get_datasets(self, train_dataset_path, valid_datasets_cfg, train_datasets_cfg=None) -> None:
        self.train_dataset_rank_sharded = self.is_distributed and self.dataset_load_mode == "eager"
        dataset_kwargs = {
            "load_mode": self.dataset_load_mode,
            "sample_sequence_length": self.sample_sequence_length,
            "max_capacity": self.dataset_max_capacity,
            "rank": self.rank if self.train_dataset_rank_sharded else 0,
            "world_size": self.world_size if self.train_dataset_rank_sharded else 1,
            **self.dataset_contact_projection,
        }
        if train_datasets_cfg is not None:
            if train_dataset_path is not None:
                raise ValueError("Configure either dataset.train_dataset_path or dataset.train_datasets, not both.")
            self.train_dataset = create_weighted_trajectory_dataset(train_datasets=train_datasets_cfg, **dataset_kwargs)
            if self.is_main_process:
                weights = {
                    name: float(weight)
                    for name, weight in zip(self.train_dataset.source_names, self.train_dataset.weights)
                }
                print_info(f"Weighted training datasets: {weights}")
        else:
            if train_dataset_path is None:
                raise ValueError("Configure dataset.train_dataset_path or dataset.train_datasets.")
            self.train_dataset = create_trajectory_dataset(hdf5_dataset_path=train_dataset_path, **dataset_kwargs)
        if valid_datasets_cfg is not None and (not self.is_distributed or self.is_main_process):
            for valid_dataset_name, valid_dataset_path in valid_datasets_cfg.items():
                self.valid_datasets[valid_dataset_name] = create_trajectory_dataset(
                    load_mode=self.dataset_load_mode,
                    sample_sequence_length=self.sample_sequence_length,
                    hdf5_dataset_path=valid_dataset_path,
                    **self.dataset_contact_projection,
                )
        self.collate_fn = None
