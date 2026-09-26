# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entry point for NeRD neural dynamics training."""

from __future__ import annotations

import copy
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_neural.envs  # noqa: F401 - registers built-in NeRD tasks
from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg, NewtonNerdManager
from isaaclab_neural.train import SequenceModelTrainer, VanillaTrainer
from isaaclab_neural.train.arguments import get_parser
from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint
from isaaclab_neural.utils.python_utils import get_time_stamp, handle_cfg_overrides, print_warning, set_random_seed

ALGORITHMS = {
    "VanillaTrainer": VanillaTrainer,
    "SequenceModelTrainer": SequenceModelTrainer,
}


def configure_distributed(args) -> None:
    """Configure CUDA device and process group when launched with torchrun."""
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = args.world_size > 1

    if not args.distributed:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA devices.")
    torch.cuda.set_device(args.local_rank)
    args.device = f"cuda:{args.local_rank}"
    timeout_seconds = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SECONDS", "3600"))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))
    print(f"[DDP] Initialized rank {args.rank}/{args.world_size} on {args.device} (pid={os.getpid()}).", flush=True)


def cleanup_distributed() -> None:
    """Tear down the torch distributed process group if it was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def configure_env(env_cfg, args) -> None:
    """Apply CLI overrides to an IsaacLab task config."""
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs") and args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
        env_cfg.sim.device = args.device


def configure_eval_terrain(env_cfg, cfg) -> dict | None:
    """Apply dataset terrain provenance before constructing the evaluation environment."""
    eval_cfg = cfg["algorithm"].get("eval", {})
    if eval_cfg.get("mode", "dataset") != "dataset":
        return None
    if not eval_cfg.get("require_terrain_context", False):
        return None
    dataset_path = eval_cfg.get("dataset_path")
    if dataset_path is None:
        return None

    from isaaclab_neural.data import read_terrain_context, set_terrain_seed

    terrain_context = read_terrain_context(dataset_path)
    if terrain_context is None:
        if eval_cfg.get("require_terrain_context", False):
            raise ValueError(
                f"Evaluation dataset {dataset_path!r} has no terrain context. Regenerate it with the current generator."
            )
        return None
    set_terrain_seed(env_cfg, int(terrain_context["seed"]))
    return terrain_context


def build_launch_cfg(env_cfg):
    """Return a Newton launch cfg while keeping the runtime env on NerdNewtonCfg."""
    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    if not isinstance(physics_cfg, NerdNewtonCfg):
        return env_cfg

    import copy

    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    launch_cfg = copy.deepcopy(env_cfg)
    launch_cfg.sim.physics = NewtonCfg(
        num_substeps=physics_cfg.num_substeps,
        debug_mode=physics_cfg.debug_mode,
        use_cuda_graph=physics_cfg.use_cuda_graph,
    )
    return launch_cfg


_INPUT_SEMANTIC_SOLVER_KEYS = (
    "name",
    "num_states_history",
    "contact_mode",
    "contact_representation",
    "contact_packing_policy",
    "contact_filter",
    "num_contacts_per_env",
    "max_contact_tokens",
    "contact_token_frame",
    "states_frame",
    "anchor_frame_step",
    "states_embedding_type",
    "prediction_type",
    "orientation_prediction_parameterization",
    "min_contact_event_threshold",
)


def _input_semantics(solver_cfg: dict) -> dict:
    """Return solver fields that define model inputs/targets, resolving omitted keys to runtime defaults."""
    defaults = NerdSolverCfg()
    semantics = {}
    for key in _INPUT_SEMANTIC_SOLVER_KEYS:
        value = solver_cfg[key] if key in solver_cfg else getattr(defaults, key)
        semantics[key] = tuple(value) if isinstance(value, list | tuple) else value
    return semantics


def validate_checkpoint_input_semantics(checkpoint_cfg: dict, cfg: dict) -> None:
    """Reject resuming a checkpoint whose input or target semantics differ from the final training config."""
    expected = _input_semantics(checkpoint_cfg["env"]["neural_solver_cfg"])
    actual = _input_semantics(cfg["env"]["neural_solver_cfg"])
    changed = {key: (expected[key], actual[key]) for key in expected if expected[key] != actual[key]}
    if cfg["inputs"] != checkpoint_cfg["inputs"]:
        changed["inputs"] = (checkpoint_cfg["inputs"], cfg["inputs"])
    if changed:
        raise ValueError(
            "Checkpoint input semantics differ from the training configuration (checkpoint, requested): "
            f"{changed}. Start a fresh model instead of resuming."
        )


def load_training_cfg(args):
    """Load training YAML and merge checkpoint/CLI overrides."""
    with open(Path(args.cfg).expanduser()) as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

    checkpoint = checkpoint_cfg = None
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(args.checkpoint, device=args.device)
        checkpoint_cfg = get_cfg_from_checkpoint(checkpoint, args.checkpoint)
        if cfg["network"].get("finger_object_normal_force_head", {}).get("enabled", False) and not (
            checkpoint_cfg["network"].get("finger_object_normal_force_head", {}).get("enabled", False)
        ):
            raise ValueError("Cannot resume filtered finger-force training from a checkpoint without that head.")
        if cfg["network"].get("robot_net_normal_force_head", {}).get("enabled", False) and not (
            checkpoint_cfg["network"].get("robot_net_normal_force_head", {}).get("enabled", False)
        ):
            raise ValueError("Cannot resume robot normal-force training from a checkpoint without that head.")
        if cfg["env"]["env_name"] != checkpoint_cfg["env"]["env_name"]:
            print_warning(
                f"Environment name in cfg {cfg['env']['env_name']} is not equal to "
                f"the environment name in checkpoint {checkpoint_cfg['env']['env_name']}"
            )
        cfg["env"]["neural_solver_cfg"].update(checkpoint_cfg["env"]["neural_solver_cfg"])
        cfg["env"]["neural_solver_cfg"].pop("use_cuda_graph", None)
        cfg["inputs"] = copy.deepcopy(checkpoint_cfg["inputs"])
        cfg["network"] = checkpoint_cfg["network"]

    handle_cfg_overrides(args.cfg_overrides, cfg)
    if checkpoint_cfg is not None:
        validate_checkpoint_input_semantics(checkpoint_cfg, cfg)
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


def train(env_cfg, args_cli) -> None:
    """Train or evaluate a resolved environment in the parent training process."""
    cfg, checkpoint = load_training_cfg(args_cli)
    validate_cfg(cfg)
    configure_env(env_cfg, args_cli)
    expected_terrain_context = configure_eval_terrain(env_cfg, cfg)
    neural_solver_cfg = dict(cfg["env"].get("neural_solver_cfg", {}))
    neural_solver_cfg.pop("use_cuda_graph", None)
    solver_cfg = NerdSolverCfg(**neural_solver_cfg)

    from isaaclab.app import launch_simulation

    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    with launch_simulation(build_launch_cfg(env_cfg), args_cli):
        import gymnasium as gym

        if expected_terrain_context is not None:
            from isaaclab_neural.data import build_terrain_context, validate_terrain_context

            actual_terrain_context = build_terrain_context(env_cfg, int(expected_terrain_context["seed"]))
            if actual_terrain_context is None:
                raise ValueError("Evaluation dataset contains terrain context but the task has no terrain generator.")
            validate_terrain_context(expected_terrain_context, actual_terrain_context)

        with (
            newton_material_binding_api_autofix(),
            NewtonNerdManager._defer_neural_model_loading(),  # pyright: ignore[reportPrivateUsage]
        ):
            env = gym.make(args_cli.task, cfg=env_cfg, device=args_cli.device, solver_cfg=solver_cfg).unwrapped
        algorithm_name = cfg["algorithm"].get("name", "VanillaTrainer")
        algorithm_cls = ALGORITHMS.get(algorithm_name)
        if algorithm_cls is None:
            raise NotImplementedError(f"Algorithm {algorithm_name} not recognized")
        if algorithm_name == "VanillaTrainer" and cfg["env"]["neural_solver_cfg"]["name"] != "NeuralSolver":
            raise ValueError("VanillaTrainer requires NeuralSolver.")

        trainer = algorithm_cls(neural_env=env, checkpoint=checkpoint, cfg=cfg, device=args_cli.device)
        env.reset()
        if args_cli.train:
            trainer.train()
        else:
            trainer.test()
        env.close()


def main() -> None:
    """Initialize the training process without side effects when imported by workers."""
    args_cli, hydra_args = get_parser().parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    try:
        configure_distributed(args_cli)
        set_random_seed(args_cli.seed)

        @hydra_task_config(args_cli.task, "")
        def run(env_cfg, _agent_cfg=None) -> None:
            train(env_cfg, args_cli)

        run()  # type: ignore[call-arg]
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
