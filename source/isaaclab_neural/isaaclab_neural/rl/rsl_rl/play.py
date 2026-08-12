# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an RSL-RL policy checkpoint on a registered NeRD-backed Isaac Lab task.

Example:
    ./isaaclab.sh -p -m isaaclab_neural.rl.rsl_rl.play \\
      --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \\
      --neural-model-path ./pre-trained_models/Anymal-C/nn/final_model.pt \\
      --checkpoint logs/rsl_rl/anymal_c_flat_nerd/<run>/model_299.pt \\
      --num_envs 16 --num_steps 200 \\
      presets=newton_mjwarp --headless
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import sys
import time

import torch

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_neural.envs  # noqa: F401 - registers NeRD gym tasks
from isaaclab_neural.rl.rsl_rl import cli_args
from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Play an RSL-RL policy on a NeRD-backed Isaac Lab task.")
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Anymal-C-NeRD-v0")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of environment steps to run.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment and agent.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Sleep to approximate real-time.")
    cli_args.add_rsl_rl_args(parser)
    cli_args.add_nerd_args(parser)

    from isaaclab_tasks.utils import add_launcher_args

    # Provides --device / --headless / etc. Do not add --device before this.
    add_launcher_args(parser)
    return parser.parse_known_args()


def reset_policy(policy, runner, dones: torch.Tensor) -> None:
    """Reset recurrent policy state when environments terminate."""
    if hasattr(policy, "reset"):
        policy.reset(dones)
        return
    policy_state = getattr(getattr(runner, "alg", None), "policy", None)
    if policy_state is None:
        policy_state = getattr(getattr(runner, "alg", None), "actor_critic", None)
    if policy_state is not None and hasattr(policy_state, "reset"):
        policy_state.reset(dones)


args_cli, hydra_args = parse_args()
sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    """Play a trained RSL-RL policy on a NeRD env."""
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_tasks.utils import get_checkpoint_path, launch_simulation

    if args_cli.checkpoint is None and not getattr(agent_cfg, "load_checkpoint", None):
        raise SystemExit("Play requires --checkpoint PATH to an RSL-RL policy file.")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    # --device comes from AppLauncher (default cuda:0).
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    solver_cfg = cli_args.apply_neural_model_to_env_cfg(env_cfg, args_cli)

    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    with launch_simulation(cli_args.build_launch_cfg(env_cfg), args_cli):
        import gymnasium as gym

        gym_kwargs = {"cfg": env_cfg, "device": agent_cfg.device}
        if solver_cfg is not None:
            gym_kwargs["solver_cfg"] = solver_cfg
        with newton_material_binding_api_autofix():
            env = gym.make(args_cli.task, **gym_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs = env.get_observations()
        dt = env.unwrapped.step_dt
        for step_id in range(args_cli.num_steps):
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, rew, dones, extras = env.step(actions)
                reset_policy(policy, runner, dones)
            reward_mean = float(torch.as_tensor(rew).float().mean())
            print(
                f"[step {step_id}] reward_mean={reward_mean:.6f}, "
                f"dones={int(torch.as_tensor(dones).sum())}, info_keys={list(extras.keys())[:5]}"
            )
            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
