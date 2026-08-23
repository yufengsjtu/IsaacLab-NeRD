# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for checkpoint controls forwarded by OSMO experiment presets."""

import argparse
from pathlib import Path
from unittest.mock import patch

from run_experiment import run_training


def test_run_training_forwards_opt_in_periodic_checkpoint_controls(tmp_path: Path) -> None:
    experiment = {
        "train_task": "test-task",
        "train_cfg": "test.yaml",
        "env_name": "test-env",
        "train_num_envs": 1,
        "num_gpus": 1,
        "save_interval": 100,
        "persist_periodic_checkpoints": True,
    }
    args = argparse.Namespace(
        train_seed=0,
        enable_wandb=True,
        wandb_project_name="test-project",
        wandb_exp_name="test-run",
        wandb_entity="",
        wandb_save_checkpoints=True,
    )

    with patch("run_experiment.subprocess.run") as run_mock:
        run_training(experiment, tmp_path, args)

    command = run_mock.call_args.args[0]
    save_interval_index = command.index("--save-interval")
    assert command[save_interval_index + 1] == "100"
    assert "--wandb-save-periodic-checkpoints" in command


def test_run_training_preserves_checkpoint_defaults_when_unset(tmp_path: Path) -> None:
    experiment = {
        "train_task": "test-task",
        "train_cfg": "test.yaml",
        "env_name": "test-env",
        "train_num_envs": 1,
        "num_gpus": 1,
    }
    args = argparse.Namespace(train_seed=0, enable_wandb=False)

    with patch("run_experiment.subprocess.run") as run_mock:
        run_training(experiment, tmp_path, args)

    command = run_mock.call_args.args[0]
    assert "--save-interval" not in command
    assert "--wandb-save-periodic-checkpoints" not in command
