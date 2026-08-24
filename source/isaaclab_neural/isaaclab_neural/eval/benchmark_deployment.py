# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark policy-closed-loop NeRD deployment without section-level synchronization."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import yaml


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse deployment benchmark arguments and preserve Hydra overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Velocity-Rough-Anymal-C-NeRD-v0")
    parser.add_argument("--checkpoint", required=True, help="NeRD dynamics checkpoint.")
    parser.add_argument("--policy-checkpoint", required=True, help="RSL-RL control-policy checkpoint.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-feature-cache", action="store_true", default=False)
    parser.add_argument("--output-json", default=None)

    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    return parser.parse_known_args()


def build_solver_cfg(args):
    """Reconstruct the runtime solver configuration from a checkpoint."""
    from isaaclab_neural.physics import NerdSolverCfg
    from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint

    checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    cfg = get_cfg_from_checkpoint(checkpoint, args.checkpoint)
    solver_values = dict(cfg["env"]["neural_solver_cfg"])
    solver_values.pop("use_cuda_graph", None)
    solver_values["neural_model_path"] = args.checkpoint
    solver_values["use_feature_cache"] = args.use_feature_cache
    return NerdSolverCfg(**solver_values)


def build_launch_cfg(env_cfg):
    """Return an upstream Newton launch configuration."""
    from isaaclab_neural.physics import NerdNewtonCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    physics_cfg = env_cfg.sim.physics
    if not isinstance(physics_cfg, NerdNewtonCfg):
        return env_cfg
    launch_cfg = copy.deepcopy(env_cfg)
    launch_cfg.sim.physics = NewtonCfg(
        num_substeps=physics_cfg.num_substeps,
        debug_mode=physics_cfg.debug_mode,
        use_cuda_graph=physics_cfg.use_cuda_graph,
    )
    return launch_cfg


def load_agent_cfg(checkpoint_path: str, device: str, seed: int) -> dict:
    """Load the RSL-RL configuration saved beside a policy checkpoint."""
    cfg_path = Path(checkpoint_path).expanduser().resolve().parent / "params" / "agent.yaml"
    with cfg_path.open() as file:
        cfg = yaml.safe_load(file)
    cfg["device"] = device
    cfg["seed"] = seed
    return cfg


args_cli, hydra_args = parse_args()

import isaaclab_tasks  # noqa: E402,F401
import isaaclab_neural.envs  # noqa: E402,F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, _agent_cfg) -> None:
    """Run a synchronized wall-clock deployment benchmark."""
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import launch_simulation

    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    if args_cli.warmup < 0 or args_cli.steps <= 0:
        raise ValueError("warmup must be non-negative and steps must be positive.")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    solver_cfg = build_solver_cfg(args_cli)
    agent_cfg = load_agent_cfg(args_cli.policy_checkpoint, args_cli.device, args_cli.seed)

    with launch_simulation(build_launch_cfg(env_cfg), args_cli):
        import gymnasium as gym

        with newton_material_binding_api_autofix():
            env = gym.make(
                args_cli.task,
                cfg=env_cfg,
                device=args_cli.device,
                solver_cfg=solver_cfg,
            )

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
        runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args_cli.device)
        runner.load(args_cli.policy_checkpoint, map_location=args_cli.device)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()

        with torch.inference_mode():
            for _ in range(args_cli.warmup):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if hasattr(policy, "reset"):
                    policy.reset(dones)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args_cli.steps):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if hasattr(policy, "reset"):
                    policy.reset(dones)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - start

        step_dt = float(env.unwrapped.step_dt)
        result = {
            "task": args_cli.task,
            "checkpoint": args_cli.checkpoint,
            "policy_checkpoint": args_cli.policy_checkpoint,
            "device": args_cli.device,
            "num_envs": args_cli.num_envs,
            "warmup": args_cli.warmup,
            "steps": args_cli.steps,
            "use_feature_cache": args_cli.use_feature_cache,
            "wall_seconds": wall_seconds,
            "wall_ms_per_step": wall_seconds * 1000.0 / args_cli.steps,
            "deployment_fps": args_cli.steps / wall_seconds,
            "env_transitions_per_second": args_cli.steps * args_cli.num_envs / wall_seconds,
            "simulated_seconds_per_wall_second": args_cli.steps * step_dt / wall_seconds,
        }
        print(json.dumps(result, indent=2))
        if args_cli.output_json:
            Path(args_cli.output_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
