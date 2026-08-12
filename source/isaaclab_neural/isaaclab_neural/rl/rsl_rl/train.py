# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an RSL-RL policy on a registered NeRD-backed Isaac Lab task.

Uses the existing :class:`~isaaclab_neural.envs.neural_env_wrapper.NerdManagerBasedRLEnv`
integration (no NewtonManager monkey-patches). Pass ``--neural-model-path`` to
point at a trained NeRD dynamics checkpoint.

Example:
    ./isaaclab.sh -p -m isaaclab_neural.rl.rsl_rl.train \\
      --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \\
      --neural-model-path ./pre-trained_models/Anymal-C/nn/final_model.pt \\
      --num_envs 64 --max_iterations 5 \\
      presets=newton_mjwarp --headless
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import sys
import time
from datetime import datetime

from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_neural.envs  # noqa: F401 - registers NeRD gym tasks
from isaaclab_neural.rl.rsl_rl import cli_args
from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train an RSL-RL agent on a NeRD-backed Isaac Lab task.")
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Anymal-C-NeRD-v0")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment and agent.")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Run training with multiple GPUs or nodes.",
    )
    cli_args.add_rsl_rl_args(parser)
    cli_args.add_nerd_args(parser)

    from isaaclab_tasks.utils import add_launcher_args

    # Provides --device / --headless / etc. Do not add --device before this.
    add_launcher_args(parser)
    return parser.parse_known_args()


args_cli, hydra_args = parse_args()
# Keep Hydra overrides (including presets=newton_mjwarp) for task resolution.
sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    """Train with RSL-RL on a NeRD env."""
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_tasks.utils import get_checkpoint_path, launch_simulation

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # --device comes from AppLauncher (default cuda:0).
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device

    env_cfg.seed = agent_cfg.seed

    if args_cli.distributed:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        global_rank = int(os.getenv("RANK", "0"))
        env_cfg.sim.device = f"cuda:{local_rank}"
        agent_cfg.device = f"cuda:{local_rank}"
        seed = agent_cfg.seed + global_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    solver_cfg = cli_args.apply_neural_model_to_env_cfg(env_cfg, args_cli)
    if args_cli.neural_model_path is None:
        default_path = getattr(getattr(env_cfg.sim.physics, "solver_cfg", None), "neural_model_path", None)
        print(
            "[WARN] --neural-model-path not set; using task default "
            f"neural_model_path={default_path!r}. Provide an explicit NeRD checkpoint for real runs."
        )

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    with launch_simulation(cli_args.build_launch_cfg(env_cfg), args_cli):
        import gymnasium as gym

        gym_kwargs = {"cfg": env_cfg, "device": agent_cfg.device}
        if solver_cfg is not None:
            gym_kwargs["solver_cfg"] = solver_cfg
        with newton_material_binding_api_autofix():
            env = gym.make(args_cli.task, **gym_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)

        if agent_cfg.resume:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        start_time = time.time()
        try:
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
