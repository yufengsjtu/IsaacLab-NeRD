# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for opt-in durable periodic NeRD checkpoints."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from isaaclab_neural.train.trainers import VanillaTrainer
from isaaclab_neural.utils.logger import Logger


class _CaptureCheckpointLogger:
    def __init__(self):
        self.wandb = object()
        self.calls = []

    def log_checkpoint(self, path: str, *, upload_now: bool = False) -> None:
        self.calls.append((Path(path).name, upload_now))


def _make_trainer(tmp_path: Path, *, upload_periodic: bool) -> VanillaTrainer:
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.neural_model = torch.nn.Linear(1, 1)
    trainer.neural_env = SimpleNamespace(robot_name="test-robot")
    trainer.cfg = {}
    trainer.model_log_dir = str(tmp_path)
    trainer.is_main_process = True
    trainer.logger = _CaptureCheckpointLogger()
    trainer.wandb_save_periodic_checkpoints = upload_periodic
    return trainer


def test_periodic_checkpoint_upload_is_opt_in(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "isaaclab_neural.train.trainers.save_checkpoint",
        lambda path, **_kwargs: Path(path).touch(),
    )
    trainer = _make_trainer(tmp_path, upload_periodic=False)

    trainer.save_model("model_epoch99")
    trainer.save_model("best_eval_model")

    assert trainer.logger.calls == [("best_eval_model.pt", False)]


def test_periodic_checkpoint_upload_requests_immediate_sync(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "isaaclab_neural.train.trainers.save_checkpoint",
        lambda path, **_kwargs: Path(path).touch(),
    )
    trainer = _make_trainer(tmp_path, upload_periodic=True)

    trainer.save_model("model_epoch99")

    assert trainer.logger.calls == [("model_epoch99.pt", True)]


def test_logger_uses_wandb_now_policy_only_when_requested(monkeypatch, tmp_path: Path) -> None:
    calls = []
    fake_wandb = SimpleNamespace(save=lambda path, **kwargs: calls.append((path, kwargs)))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    logger = Logger()
    logger.wandb = object()
    checkpoint = tmp_path / "model_epoch99.pt"
    checkpoint.touch()

    logger.log_checkpoint(str(checkpoint), upload_now=True)

    assert calls == [
        (
            str(checkpoint),
            {"base_path": str(tmp_path), "policy": "now"},
        )
    ]
