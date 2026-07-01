# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entry point for NeRD neural dynamics training."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import yaml

import isaaclab_neural.envs  # noqa: F401 - registers built-in NeRD tasks
from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.train.arguments import get_parser
from isaaclab_neural.train import MultiStepTrainer, MultiStepTrainerNew, SequenceModelTrainer, VanillaTrainer
from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint
from isaaclab_neural.utils.python_utils import get_time_stamp, handle_cfg_overrides, print_warning, set_random_seed
from isaaclab_tasks.utils.hydra import hydra_task_config


ALGORITHMS = {
    "VanillaTrainer": VanillaTrainer,
    "SequenceModelTrainer": SequenceModelTrainer,
    "MultiStepTrainer": MultiStepTrainer,
    "MultiStepTrainerNew": MultiStepTrainerNew,
}


def configure_env(env_cfg, args) -> None:
    """Apply CLI overrides to an IsaacLab task config."""
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs") and args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
        env_cfg.sim.device = args.device


def build_launch_cfg(env_cfg):
    """Return a Newton launch cfg while keeping the runtime env on NerdNewtonCfg."""
    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    if not isinstance(physics_cfg, NerdNewtonCfg):
        return env_cfg

    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    import copy

    launch_cfg = copy.deepcopy(env_cfg)
    launch_cfg.sim.physics = NewtonCfg(
        num_substeps=physics_cfg.num_substeps,
        debug_mode=physics_cfg.debug_mode,
        use_cuda_graph=physics_cfg.use_cuda_graph,
    )
    return launch_cfg


def load_training_cfg(args):
    """Load training YAML and merge checkpoint/CLI overrides."""
    with open(Path(args.cfg).expanduser(), "r") as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(args.checkpoint, device=args.device)
        checkpoint_cfg = get_cfg_from_checkpoint(checkpoint, args.checkpoint)
        if cfg["env"]["env_name"] != checkpoint_cfg["env"]["env_name"]:
            print_warning(
                f"Environment name in cfg {cfg['env']['env_name']} is not equal to "
                f"the environment name in checkpoint {checkpoint_cfg['env']['env_name']}"
            )
        cfg["env"]["neural_solver_cfg"].update(checkpoint_cfg["env"]["neural_solver_cfg"])
        cfg["inputs"] = checkpoint_cfg["inputs"]
        cfg["network"] = checkpoint_cfg["network"]

    handle_cfg_overrides(args.cfg_overrides, cfg)
    if not args.no_time_stamp:
        timestamp = get_time_stamp()
        args.logdir = os.path.join(args.logdir, timestamp)
        if args.enable_wandb and args.wandb_exp_name is not None:
            args.wandb_exp_name = f"{args.wandb_exp_name}/{timestamp}"

    if args.num_envs is not None:
        cfg["env"]["num_envs"] = args.num_envs
    cfg["env"]["render"] = args.render
    cfg["algorithm"]["seed"] = args.seed
    cfg["algorithm"]["update_dataset_statistics"] = args.update_dataset_statistics
    args.train = not args.test
    cfg["cli"] = vars(args).copy()
    cfg["cli"]["train"] = args.train
    cfg["cli"].pop("num_envs", None)
    cfg["cli"].pop("seed", None)
    return cfg, checkpoint


def validate_cfg(cfg) -> None:
    """Validate solver/network consistency for common NeRD training configs."""
    neural_solver_name = cfg["env"]["neural_solver_cfg"]["name"]
    if "transformer" in cfg["network"]:
        if neural_solver_name != "TransformerNeuralSolver":
            raise ValueError("Transformer network requires TransformerNeuralSolver.")
        if cfg["env"]["neural_solver_cfg"].get("num_states_history") != cfg["algorithm"]["sample_sequence_length"]:
            raise ValueError("num_states_history must equal sample_sequence_length for Transformer training.")
    elif "rnn" in cfg["network"]:
        if neural_solver_name != "RNNNeuralSolver":
            raise ValueError("RNN network requires RNNNeuralSolver.")
        if cfg["env"]["neural_solver_cfg"].get("reset_seq_length", 1) != cfg["algorithm"]["sample_sequence_length"]:
            raise ValueError("reset_seq_length must equal sample_sequence_length for RNN training.")


args_cli, hydra_args = get_parser().parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

@hydra_task_config(args_cli.task, "")
def main(env_cfg, _agent_cfg=None) -> None:
    cfg, checkpoint = load_training_cfg(args_cli)
    validate_cfg(cfg)
    configure_env(env_cfg, args_cli)
    solver_cfg = NerdSolverCfg(**cfg["env"].get("neural_solver_cfg", {}))

    from isaaclab_tasks.utils import launch_simulation

    with launch_simulation(build_launch_cfg(env_cfg), args_cli):
        import gymnasium as gym

        env = gym.make(args_cli.task, cfg=env_cfg, device=args_cli.device, solver_cfg=solver_cfg).unwrapped
        env.reset()
        algorithm_name = cfg["algorithm"].get("name", "VanillaTrainer")
        algorithm_cls = ALGORITHMS.get(algorithm_name)
        if algorithm_cls is None:
            raise NotImplementedError(f"Algorithm {algorithm_name} not recognized")
        if algorithm_name == "VanillaTrainer" and cfg["env"]["neural_solver_cfg"]["name"] != "NeuralSolver":
            raise ValueError("VanillaTrainer requires NeuralSolver.")

        trainer = algorithm_cls(neural_env=env, checkpoint=checkpoint, cfg=cfg, device=args_cli.device)
        if args_cli.train:
            trainer.train()
        else:
            trainer.test()
        env.close()


if __name__ == "__main__":
    set_random_seed(args_cli.seed)
    main()  # type: ignore[call-arg]
