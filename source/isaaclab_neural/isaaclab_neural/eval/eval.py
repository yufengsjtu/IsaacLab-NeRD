# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic NeRD evaluation runner for registered IsaacLab tasks.

Configuration precedence:
    1. The registered task provides the default ``NerdSolverCfg``.
    2. ``--checkpoint`` replaces it with the NeRD checkpoint's embedded solver cfg.
    3. CLI solver flags override the active solver cfg.

Policy checkpoints use the adjacent ``params/agent.yaml`` when present so RSL-RL
inference matches training-time runner settings such as ``clip_actions``.

``build_launch_cfg()`` is a launch-only helper: it gives
``launch_simulation()`` an upstream ``NewtonCfg`` for backend/visualizer detection,
while the actual environment is still created with ``NerdNewtonCfg``.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata as metadata
import sys
from collections.abc import Mapping
from pathlib import Path

import isaaclab_neural.envs  # noqa: F401 - registers built-in NeRD eval tasks
import torch
from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint

from isaaclab_tasks.utils.hydra import hydra_task_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Evaluate a registered IsaacLab NeRD task.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-NeRD-v0", help="Registered gym task id.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional NeRD checkpoint override.")
    parser.add_argument("--policy-checkpoint", type=str, default=None, help="Optional RSL-RL policy checkpoint.")
    parser.add_argument(
        "--policy-agent",
        type=str,
        default="rsl_rl_cfg_entry_point",
        help="Gym registry key for the RSL-RL agent config.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs.")
    parser.add_argument("--num-steps", type=int, default=8, help="Number of env steps to run.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--contact-mode", choices=["fixed_ground", "newton_native"], default=None)
    parser.add_argument("--num-contacts-per-env", type=int, default=None)
    parser.add_argument(
        "--use-cuda-graph",
        choices=["true", "false"],
        default=None,
        help=(
            "Override the upstream Newton CUDA graph preference. Has no effect for NeRD "
            "neural solvers, which always run physics eagerly."
        ),
    )
    parser.add_argument("--video", action="store_true", default=False, help="Record eval rollout video.")
    parser.add_argument("--video-length", "--video_length", dest="video_length", type=int, default=400)
    parser.add_argument("--video-interval", "--video_interval", dest="video_interval", type=int, default=2000)
    parser.add_argument("--video-dir", "--video_dir", dest="video_dir", type=str, default="./videos/eval")

    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    return parser.parse_known_args()


def apply_solver_overrides(solver_cfg: NerdSolverCfg, args: argparse.Namespace) -> None:
    if args.contact_mode is not None:
        solver_cfg.contact_mode = args.contact_mode
    if args.num_contacts_per_env is not None:
        solver_cfg.num_contacts_per_env = args.num_contacts_per_env


def apply_cuda_graph_override(
    env_cfg,
    args: argparse.Namespace,
    *,
    legacy_use_cuda_graph: bool | None = None,
) -> None:
    """Apply ``--use-cuda-graph`` to the physics config, matching upstream ``NewtonCfg``."""
    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    if not isinstance(physics_cfg, NerdNewtonCfg):
        return

    if args.use_cuda_graph is not None:
        physics_cfg.use_cuda_graph = args.use_cuda_graph == "true"
    elif legacy_use_cuda_graph is not None:
        physics_cfg.use_cuda_graph = legacy_use_cuda_graph


def build_solver_cfg(args: argparse.Namespace) -> tuple[NerdSolverCfg | None, bool | None]:
    if args.checkpoint is None:
        return None, None

    checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    cfg = get_cfg_from_checkpoint(checkpoint, args.checkpoint)
    neural_solver_cfg = dict(cfg["env"]["neural_solver_cfg"])
    legacy_use_cuda_graph = neural_solver_cfg.pop("use_cuda_graph", None)
    neural_solver_cfg["neural_model_path"] = args.checkpoint
    solver_cfg = NerdSolverCfg(**neural_solver_cfg)
    apply_solver_overrides(solver_cfg, args)
    return solver_cfg, legacy_use_cuda_graph


def configure_env(env_cfg, args: argparse.Namespace) -> None:
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
        if getattr(args, "device", None) is None:
            args.device = env_cfg.sim.device
        else:
            env_cfg.sim.device = args.device

    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    if args.checkpoint is None and isinstance(physics_cfg, NerdNewtonCfg):
        apply_solver_overrides(physics_cfg.solver_cfg, args)


def build_launch_cfg(env_cfg):
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


def zero_action(env) -> torch.Tensor:
    shape = env.action_space.shape
    if len(shape) == 1:
        shape = (getattr(env.unwrapped, "num_envs", getattr(env, "num_envs", 1)), shape[0])
    device = getattr(env.unwrapped, "device", getattr(env, "device", "cpu"))
    return torch.zeros(shape, device=device, dtype=torch.float32)


def summarize_step(result) -> str:
    if not isinstance(result, tuple) or len(result) < 5:
        return f"result_type={type(result).__name__}"

    obs, rew, terminated, truncated, info = result
    obs_summary = f"obs_keys={list(obs.keys())[:5]}" if isinstance(obs, Mapping) else f"obs_type={type(obs).__name__}"
    return (
        f"{obs_summary}, reward_mean={float(torch.as_tensor(rew).float().mean()):.6f}, "
        f"terminated={int(torch.as_tensor(terminated).sum())}, "
        f"truncated={int(torch.as_tensor(truncated).sum())}, info_keys={list(info.keys())[:5]}"
    )


def load_saved_agent_cfg_for_policy(checkpoint_path: str, fallback_agent_cfg, *, device: str, seed: int):
    """Load the RSL-RL runner config saved next to a policy checkpoint when available."""
    agent_cfg_path = Path(checkpoint_path).expanduser().resolve().parent / "params" / "agent.yaml"
    if not agent_cfg_path.exists():
        return fallback_agent_cfg

    import yaml

    with agent_cfg_path.open("r") as f:
        agent_cfg = yaml.safe_load(f)
    if not isinstance(agent_cfg, Mapping):
        raise ValueError(f"Expected a mapping in saved RSL-RL agent config: {agent_cfg_path}")

    agent_cfg = copy.deepcopy(dict(agent_cfg))
    agent_cfg["device"] = device
    agent_cfg["seed"] = seed
    print(f"[policy] using saved RSL-RL agent config: {agent_cfg_path}")
    return agent_cfg


def normalize_runner_cfg(agent_cfg, *, device: str):
    if isinstance(agent_cfg, Mapping):
        runner_cfg = copy.deepcopy(dict(agent_cfg))
        runner_cfg["device"] = device
        return (
            runner_cfg,
            runner_cfg.get("class_name", "OnPolicyRunner"),
            runner_cfg.get("clip_actions"),
            runner_cfg.get("device", device),
        )

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    runner_cfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(vars(agent_cfg))
    return (
        runner_cfg,
        getattr(agent_cfg, "class_name", "OnPolicyRunner"),
        getattr(agent_cfg, "clip_actions", None),
        getattr(agent_cfg, "device", device),
    )


def build_policy_env(env, agent_cfg, checkpoint_path: str, device: str):
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    runner_cfg, class_name, clip_actions, runner_device = normalize_runner_cfg(agent_cfg, device=device)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
    runner_cls = {"OnPolicyRunner": OnPolicyRunner, "DistillationRunner": DistillationRunner}.get(class_name)
    if runner_cls is None:
        raise ValueError(f"Unsupported RSL-RL runner class: {class_name}")

    print(f"[policy] loading RSL-RL checkpoint: {checkpoint_path}")
    runner = runner_cls(wrapped_env, runner_cfg, log_dir=None, device=runner_device)
    runner.load(checkpoint_path, map_location=device)
    return wrapped_env, runner.get_inference_policy(device=wrapped_env.unwrapped.device), runner


def reset_policy(policy, runner, dones: torch.Tensor) -> None:
    if hasattr(policy, "reset"):
        policy.reset(dones)
        return

    policy_state = getattr(getattr(runner, "alg", None), "policy", None)
    if policy_state is None:
        policy_state = getattr(getattr(runner, "alg", None), "actor_critic", None)
    if policy_state is not None and hasattr(policy_state, "reset"):
        policy_state.reset(dones)


def wrap_record_video(env, args: argparse.Namespace):
    """Wrap the environment with Gymnasium's video recorder when requested."""
    if not args.video:
        return env
    if args.video_length <= 0:
        raise ValueError("--video-length must be positive.")
    if args.video_interval <= 0:
        raise ValueError("--video-interval must be positive.")

    import gymnasium as gym

    video_folder = str(Path(args.video_dir).expanduser())
    print(f"[video] recording eval rollout to: {video_folder}")
    return gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        step_trigger=lambda step: step % args.video_interval == 0,
        video_length=args.video_length,
        disable_logger=True,
    )


def run_zero_action(env, args: argparse.Namespace) -> None:
    print(f"[reset] type={type(env.reset()).__name__}")
    action = zero_action(env)
    print(
        f"[eval] task={args.task}, num_envs={args.num_envs}, steps={args.num_steps}, "
        f"action_shape={tuple(action.shape)}, action_source=zero"
    )
    for step_id in range(args.num_steps):
        print(f"[step {step_id}] {summarize_step(env.step(action))}")
    env.close()


def run_policy(env, agent_cfg, args: argparse.Namespace) -> None:
    agent_cfg = load_saved_agent_cfg_for_policy(args.policy_checkpoint, agent_cfg, device=args.device, seed=args.seed)
    if agent_cfg is None:
        raise ValueError(f"No RSL-RL agent config found for entry point: {args.policy_agent}")

    env, policy, runner = build_policy_env(env, agent_cfg, args.policy_checkpoint, args.device)
    obs = env.get_observations()
    print(f"[eval] task={args.task}, num_envs={args.num_envs}, steps={args.num_steps}, action_source=rsl_rl")
    for step_id in range(args.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, extras = env.step(actions)
            reset_policy(policy, runner, dones)
        print(
            f"[step {step_id}] reward_mean={float(torch.as_tensor(rew).float().mean()):.6f}, "
            f"dones={int(torch.as_tensor(dones).sum())}, info_keys={list(extras.keys())[:5]}"
        )
    env.close()


args_cli, hydra_args = parse_args()
sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(args_cli.task, args_cli.policy_agent if args_cli.policy_checkpoint is not None else "")
def main(env_cfg, agent_cfg) -> None:
    configure_env(env_cfg, args_cli)
    solver_cfg, legacy_use_cuda_graph = build_solver_cfg(args_cli)
    apply_cuda_graph_override(env_cfg, args_cli, legacy_use_cuda_graph=legacy_use_cuda_graph)

    from isaaclab_tasks.utils import launch_simulation

    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    with launch_simulation(build_launch_cfg(env_cfg), args_cli):
        import gymnasium as gym

        gym_kwargs = {"cfg": env_cfg, "device": args_cli.device}
        if args_cli.video:
            gym_kwargs["render_mode"] = "rgb_array"
        if solver_cfg is not None:
            gym_kwargs["solver_cfg"] = solver_cfg
        with newton_material_binding_api_autofix():
            env = wrap_record_video(gym.make(args_cli.task, **gym_kwargs), args_cli)

        if args_cli.policy_checkpoint is None:
            run_zero_action(env, args_cli)
        else:
            run_policy(env, agent_cfg, args_cli)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
