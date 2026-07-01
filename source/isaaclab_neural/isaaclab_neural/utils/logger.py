# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight training logger for TensorBoard and optional Weights & Biases."""

from __future__ import annotations

from typing import Any


class Logger:
    """Small logger used by NeRD trainers."""

    def __init__(self):
        self.tensorboard_writer = None
        self.wandb = None
        self.wandb_logs: dict[str, Any] = {}
        self.current_step = 0

    def init_tensorboard(self, summary_log_dir: str) -> None:
        """Initialize TensorBoard logging."""
        from torch.utils.tensorboard import SummaryWriter

        self.tensorboard_writer = SummaryWriter(summary_log_dir)

    def init_wandb(self, wandb_project: str, wandb_name: str) -> None:
        """Initialize Weights & Biases logging."""
        import wandb

        self.wandb = wandb.init(project=wandb_project, name=wandb_name)

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
