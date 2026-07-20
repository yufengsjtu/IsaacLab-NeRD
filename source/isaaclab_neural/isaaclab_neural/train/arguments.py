# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line arguments for NeRD model training."""

from __future__ import annotations

import argparse


def get_parser() -> argparse.ArgumentParser:
    """Create the NeRD training argument parser."""
    parser = argparse.ArgumentParser(description="Train a NeRD neural dynamics model.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-NeRD-v0", help="Registered gym task id.")
    parser.add_argument("--cfg", type=str, default="./cfg/Cartpole/transformer.yaml", help="Training config YAML.")
    parser.add_argument("--test", action="store_true", help="Run trainer test path instead of training.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Restore model/training state from checkpoint.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override number of envs in task config.")
    parser.add_argument("--update-dataset-statistics", action="store_true", help="Recompute dataset RMS statistics.")
    parser.add_argument("--logdir", type=str, default="./trained_models/ppo/", help="Training log directory.")
    parser.add_argument("--no-time-stamp", action="store_true", help="Do not append timestamp to log directory.")
    parser.add_argument("--eval-interval", type=int, default=5, help="Override rollout eval interval.")
    parser.add_argument("--save-interval", type=int, default=200, help="Epoch interval between checkpoints.")
    parser.add_argument("--log-interval", type=int, default=1, help="Epoch interval between logs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--render", action="store_true", default=False, help="Render during env creation/eval.")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project-name", type=str, default="nerd-newton")
    parser.add_argument("--wandb-exp-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity or team name.")
    parser.add_argument(
        "--wandb-save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload best checkpoints to the active W&B run.",
    )
    parser.add_argument("--skip-check-log-override", action="store_true")
    parser.add_argument("--cfg-overrides", default="", type=str, help="Pairs of dotted config keys and values.")

    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    return parser
