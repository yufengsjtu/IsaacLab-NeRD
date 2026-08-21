# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight training logger for TensorBoard and optional Weights & Biases."""

from __future__ import annotations

import os
from typing import Any


class Logger:
    """Small logger used by NeRD trainers."""

    def __init__(self):
        self.tensorboard_writer = None
        self.wandb = None
        self.wandb_logs: dict[str, Any] = {}
        self.current_step = 0
        self.save_checkpoints = True

    def init_tensorboard(self, summary_log_dir: str) -> None:
        """Initialize TensorBoard logging."""
        from torch.utils.tensorboard import SummaryWriter

        self.tensorboard_writer = SummaryWriter(summary_log_dir)

    def init_wandb(
        self,
        wandb_project: str,
        wandb_name: str | None,
        *,
        wandb_entity: str | None = None,
        config: dict[str, Any] | None = None,
        save_checkpoints: bool = True,
        system_stats_interval_seconds: float | None = None,
    ) -> None:
        """Initialize Weights & Biases logging.

        When enabled, W&B captures stdout/stderr from this process and can upload
        selected checkpoint files through :meth:`log_checkpoint`.
        """
        import wandb

        self.save_checkpoints = save_checkpoints
        settings_kwargs: dict[str, Any] = {"console": "wrap"}
        if system_stats_interval_seconds is not None:
            settings_kwargs["x_stats_sampling_interval"] = system_stats_interval_seconds
        init_kwargs: dict[str, Any] = {
            "project": wandb_project,
            "name": wandb_name,
            "settings": wandb.Settings(**settings_kwargs),
        }
        if wandb_entity:
            init_kwargs["entity"] = wandb_entity
        if config:
            init_kwargs["config"] = config
        self.wandb = wandb.init(**init_kwargs)

    def init_epoch(self, epoch: int) -> None:
        """Start collecting logs for one epoch."""
        self.current_step = epoch
        self.wandb_logs = {}

    def add_scalar(self, name: str, value: Any, step: int) -> None:
        """Log one scalar."""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(name, value, step)
        if self.wandb:
            self.wandb_logs[name] = value

    def log_text_file(self, path: str, key: str) -> None:
        """Upload a text artifact to the active W&B run."""
        if not self.wandb:
            return
        import wandb

        artifact = wandb.Artifact(key, type="log")
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def log_checkpoint(self, path: str) -> None:
        """Upload a checkpoint file to the active W&B run."""
        if not self.wandb or not self.save_checkpoints:
            return
        import wandb

        wandb.save(path, base_path=os.path.dirname(path))

    def flush(self) -> None:
        """Flush buffered logs."""
        if self.tensorboard_writer:
            self.tensorboard_writer.flush()
        if self.wandb and len(self.wandb_logs) > 0:
            import wandb

            wandb.log(self.wandb_logs, step=self.current_step)

    def finish(self) -> None:
        """Close external logging sessions."""
        if self.wandb:
            import wandb

            wandb.finish()
