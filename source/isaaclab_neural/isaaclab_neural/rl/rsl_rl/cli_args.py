# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI helpers for NeRD RSL-RL train/play entry points."""

from __future__ import annotations

import argparse
import copy
import random
from typing import TYPE_CHECKING

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add RSL-RL arguments to the parser."""
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    arg_group.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment folder where logs will be stored.",
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="RSL-RL policy checkpoint to resume from (train) or play.",
    )
    arg_group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard", "neptune"},
        help="Logger module to use.",
    )
    arg_group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Name of the logging project when using wandb or neptune.",
    )
    arg_group.add_argument(
        "--clip-actions",
        type=float,
        default=None,
        help="Override action clipping passed to RslRlVecEnvWrapper.",
    )


def add_nerd_args(parser: argparse.ArgumentParser) -> None:
    """Add NeRD dynamics checkpoint / solver override arguments."""
    group = parser.add_argument_group("nerd", description="NeRD physics options.")
    group.add_argument(
        "--neural-model-path",
        type=str,
        default=None,
        help=(
            "Path to a NeRD dynamics checkpoint (.pt). When set, restores the embedded "
            "NerdSolverCfg and points neural_model_path at this file."
        ),
    )
    group.add_argument("--contact-mode", choices=["fixed_ground", "newton_native"], default=None)
    group.add_argument("--num-contacts-per-env", type=int, default=None)


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Update an RSL-RL agent config from parsed CLI args."""
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    if getattr(args_cli, "clip_actions", None) is not None:
        agent_cfg.clip_actions = args_cli.clip_actions
    return agent_cfg


def apply_solver_overrides(solver_cfg: NerdSolverCfg, args: argparse.Namespace) -> None:
    """Apply CLI contact overrides onto a solver config."""
    if args.contact_mode is not None:
        solver_cfg.contact_mode = args.contact_mode
    if args.num_contacts_per_env is not None:
        solver_cfg.num_contacts_per_env = args.num_contacts_per_env


def build_solver_cfg_from_neural_model(
    args: argparse.Namespace,
) -> tuple[NerdSolverCfg | None, bool | None]:
    """Build a NerdSolverCfg from ``--neural-model-path`` when provided."""
    if args.neural_model_path is None:
        return None, None

    checkpoint = load_checkpoint(args.neural_model_path, device="cpu")
    cfg = get_cfg_from_checkpoint(checkpoint, args.neural_model_path)
    neural_solver_cfg = dict(cfg["env"]["neural_solver_cfg"])
    legacy_use_cuda_graph = neural_solver_cfg.pop("use_cuda_graph", None)
    neural_solver_cfg["neural_model_path"] = args.neural_model_path
    solver_cfg = NerdSolverCfg(**neural_solver_cfg)
    apply_solver_overrides(solver_cfg, args)
    return solver_cfg, legacy_use_cuda_graph


def apply_neural_model_to_env_cfg(env_cfg, args: argparse.Namespace) -> NerdSolverCfg | None:
    """Attach NeRD solver overrides to the env cfg and return gym.make solver_cfg."""
    from isaaclab_neural.envs.anymal_nerd_env import apply_contact_mode_mdp

    solver_cfg, legacy_use_cuda_graph = build_solver_cfg_from_neural_model(args)
    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    if not isinstance(physics_cfg, NerdNewtonCfg):
        raise ValueError("NeRD RL training requires env_cfg.sim.physics to be a NerdNewtonCfg.")

    if solver_cfg is not None:
        physics_cfg.solver_cfg = solver_cfg
        if legacy_use_cuda_graph is not None:
            physics_cfg.use_cuda_graph = bool(legacy_use_cuda_graph)
    else:
        apply_solver_overrides(physics_cfg.solver_cfg, args)
        if args.neural_model_path is not None:
            physics_cfg.solver_cfg.neural_model_path = args.neural_model_path

    physics_cfg.use_cuda_graph = False
    # Re-sync contact rewards after CLI ``--contact-mode`` overrides.
    apply_contact_mode_mdp(env_cfg)
    return solver_cfg


def build_launch_cfg(env_cfg):
    """Return a launch cfg with upstream NewtonCfg for backend detection."""
    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    if not isinstance(physics_cfg, NerdNewtonCfg):
        return env_cfg

    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    launch_cfg = copy.deepcopy(env_cfg)
    launch_cfg.sim.physics = NewtonCfg(
        num_substeps=physics_cfg.num_substeps,
        debug_mode=physics_cfg.debug_mode,
        use_cuda_graph=physics_cfg.use_cuda_graph,
    )
    return launch_cfg
