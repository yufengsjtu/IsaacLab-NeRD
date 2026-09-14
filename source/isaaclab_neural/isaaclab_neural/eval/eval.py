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
import sys
from collections.abc import Mapping
from pathlib import Path

import isaaclab_neural.envs  # noqa: F401 - registers built-in NeRD eval tasks
import torch
from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.physics.nerd_solver_cfg import normalize_contact_mode
from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint
from isaaclab_neural.utils.checkpoint import load_saved_agent_cfg_for_policy as _load_saved_agent_cfg_for_policy
from isaaclab_neural.utils.checkpoint import normalize_runner_cfg as _normalize_runner_cfg

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
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help=(
            "Record an eval mp4. NeRD tasks capture Newton GL (not Kit/RTX). "
            "Recording-only runs stay headless; pass --viz newton to open a live window. "
            "Pass --viz kit only if Kit recording is required."
        ),
    )
    parser.add_argument(
        "--video-length",
        "--video_length",
        dest="video_length",
        type=int,
        default=400,
        help="Env steps captured per clip. Clips shorter than this are flushed on env.close().",
    )
    parser.add_argument(
        "--video-interval",
        "--video_interval",
        dest="video_interval",
        type=int,
        default=0,
        help="Start a new clip every N env steps. 0 records a single clip from the first step.",
    )
    parser.add_argument("--video-dir", "--video_dir", dest="video_dir", type=str, default="./videos/eval")
    parser.add_argument(
        "--plot-rewards",
        action="store_true",
        default=False,
        help="Save the per-step mean reward curve after evaluation.",
    )
    parser.add_argument(
        "--reward-plot-path",
        type=str,
        default=None,
        help="Reward plot output path. Defaults to <video-dir>/rewards.png.",
    )

    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    _alias_vis_as_viz(parser)
    return parser.parse_known_args()


def _alias_vis_as_viz(parser: argparse.ArgumentParser) -> None:
    """Accept ``--vis`` as ``--viz``. Hydra would otherwise swallow the unknown flag."""
    for action in parser._actions:
        if "--viz" in action.option_strings and "--vis" not in action.option_strings:
            action.option_strings = [*action.option_strings, "--vis"]
            parser._option_string_actions["--vis"] = action
            return


def apply_solver_overrides(solver_cfg: NerdSolverCfg, args: argparse.Namespace) -> None:
    if args.contact_mode is not None:
        solver_cfg.contact_mode = normalize_contact_mode(args.contact_mode)
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


_NEWTON_CAPTURE_VISUALIZERS = frozenset({"newton_gl", "newton", "newton_rtx"})
_NO_CAPTURE_VISUALIZERS = frozenset({"rerun", "viser"})


def configure_eval_video(env_cfg, args: argparse.Namespace) -> None:
    """Enable Isaac Lab 3 step-driven video recording when ``--video`` is set.

    Gymnasium ``RecordVideo`` plus ``render_mode=rgb_array`` no longer captures
    frames. Recording is configured on ``env_cfg.video_recorders`` before launch.

    NeRD/Newton eval records from the Newton GL visualizer. The Isaac Lab RL
    ``--video`` helpers inject Kit so the launcher includes RTX; that Kit
    startup crashed on a hybrid NVIDIA+Intel Vulkan host. Pass ``--viz kit``
    only if Kit recording is required.
    """
    if not args.video:
        return
    if args.video_length <= 0:
        raise ValueError("--video-length must be positive.")
    if args.video_interval < 0:
        raise ValueError("--video-interval must be >= 0.")

    video_dir = str(Path(args.video_dir).expanduser())
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    viz_explicitly_none = (
        getattr(args, "visualizer_explicit", False) and getattr(args, "visualizer", "not_none") is None
    )
    if viz_explicitly_none:
        raise ValueError(
            "--video is not compatible with --viz none: there is no active visualizer to record from. "
            "Remove --viz none, or pass --viz newton_gl (NeRD default) or --viz kit."
        )

    requested = _requested_visualizers(args)
    user_requested_window = bool(getattr(args, "visualizer_explicit", False)) and bool(requested)
    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    use_newton_capture = (
        isinstance(physics_cfg, NerdNewtonCfg)
        and "kit" not in requested
        and not any(name in _NO_CAPTURE_VISUALIZERS for name in requested)
    )
    if use_newton_capture:
        source_type = next((name for name in requested if name in _NEWTON_CAPTURE_VISUALIZERS), "newton_gl")
        _configure_newton_capture_eval_video(env_cfg, args, video_dir, source_type, headless=not user_requested_window)
        return

    from isaaclab_rl.entrypoints.common import (
        apply_video_recording,
        enable_cameras_for_video,
        pre_launch_video_config,
    )

    enable_cameras_for_video(args)
    pre_launch_video_config(env_cfg, args_cli=args)
    apply_video_recording(env_cfg, video_dir, args, subdir="eval")
    for recorder_cfg in getattr(env_cfg, "video_recorders", []) or []:
        recorder_cfg.output_dir = video_dir
    print(f"[video] recording eval rollout to: {video_dir}")


def _requested_visualizers(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "visualizer", None)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [str(part).strip() for part in raw if str(part).strip()]


def _canonical_newton_visualizer(name: str) -> str:
    if name == "newton":
        return "newton_gl"
    return name


def _configure_newton_capture_eval_video(
    env_cfg, args: argparse.Namespace, video_dir: str, source_type: str, *, headless: bool
) -> None:
    """Record from Newton GL/RTX so eval does not launch Kit.

    ``headless=True`` writes the mp4 without a window. Pass ``--viz newton`` to
    set ``headless=False`` and open the live viewer while recording.
    """
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    canonical = _canonical_newton_visualizer(source_type)
    args.enable_cameras = False
    args.visualizer = [canonical]
    args.visualizer_explicit = True

    sim_cfg = env_cfg.sim
    default_viz = getattr(sim_cfg, "default_visualizer_cfg", None)
    eye = getattr(default_viz, "eye", (3.5, 3.5, 3.5))
    lookat = getattr(default_viz, "lookat", (0.0, 0.0, 0.0))
    if not isinstance(getattr(sim_cfg, "visualizer_cfgs", None), list):
        sim_cfg.visualizer_cfgs = []
    existing = next(
        (cfg for cfg in sim_cfg.visualizer_cfgs if getattr(cfg, "visualizer_type", None) == canonical),
        None,
    )
    if existing is None:
        viz_cls = NewtonRTXVisualizerCfg if canonical == "newton_rtx" else NewtonGLVisualizerCfg
        sim_cfg.visualizer_cfgs.append(viz_cls(headless=headless, eye=eye, lookat=lookat))
    else:
        existing.headless = headless

    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source=f"visualizer:{canonical}",
            output_dir=video_dir,
            video_length=args.video_length,
            video_interval=args.video_interval,
            output_filename_prefix="eval",
        )
    ]
    window = "off" if headless else "on"
    print(f"[video] recording Newton {canonical} eval rollout to: {video_dir} (window={window})")


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
    return _load_saved_agent_cfg_for_policy(checkpoint_path, fallback_agent_cfg, device=device, seed=seed)


def normalize_runner_cfg(agent_cfg, *, device: str):
    return _normalize_runner_cfg(agent_cfg, device=device)


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
    """No-op kept for import compatibility.

    Recording is configured on ``env_cfg.video_recorders`` before ``gym.make``.
    """
    del args
    return env


def save_reward_plot(reward_means: list[float], args: argparse.Namespace) -> None:
    """Save the per-step mean reward and its moving average."""
    if not args.plot_rewards:
        return
    if not reward_means:
        raise ValueError("Cannot plot rewards because evaluation produced no steps.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = (
        Path(args.reward_plot_path).expanduser()
        if args.reward_plot_path is not None
        else Path(args.video_dir).expanduser() / "rewards.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = list(range(len(reward_means)))
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(steps, reward_means, linewidth=1.0, alpha=0.55, label="Mean reward")
    if len(reward_means) > 1:
        window = min(20, len(reward_means))
        moving_average = torch.as_tensor(reward_means).unfold(0, window, 1).mean(dim=1).tolist()
        moving_steps = list(range(window - 1, len(reward_means)))
        axis.plot(moving_steps, moving_average, linewidth=2.0, label=f"{window}-step moving average")
    axis.set(xlabel="Environment step", ylabel="Mean reward", title=f"Evaluation reward: {args.task}")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"[reward-plot] saved to: {output_path}")


def run_zero_action(env, args: argparse.Namespace) -> None:
    print(f"[reset] type={type(env.reset()).__name__}")
    action = zero_action(env)
    reward_means = []
    print(
        f"[eval] task={args.task}, num_envs={args.num_envs}, steps={args.num_steps}, "
        f"action_shape={tuple(action.shape)}, action_source=zero"
    )
    for step_id in range(args.num_steps):
        result = env.step(action)
        reward_means.append(float(torch.as_tensor(result[1]).float().mean()))
        print(f"[step {step_id}] {summarize_step(result)}")
    save_reward_plot(reward_means, args)
    env.close()


def run_policy(env, agent_cfg, args: argparse.Namespace) -> None:
    agent_cfg = load_saved_agent_cfg_for_policy(args.policy_checkpoint, agent_cfg, device=args.device, seed=args.seed)
    if agent_cfg is None:
        raise ValueError(f"No RSL-RL agent config found for entry point: {args.policy_agent}")

    env, policy, runner = build_policy_env(env, agent_cfg, args.policy_checkpoint, args.device)
    obs = env.get_observations()
    reward_means = []
    print(f"[eval] task={args.task}, num_envs={args.num_envs}, steps={args.num_steps}, action_source=rsl_rl")
    for step_id in range(args.num_steps):
        # Only the policy forward belongs under inference_mode. env.step()
        # clones Transformer history and later index_copy_s those rows on
        # Reach/Lift success resets; mutating inference tensors raises
        # "Inplace update to inference tensor".
        with torch.inference_mode():
            actions = policy(obs)
        obs, rew, dones, extras = env.step(actions)
        reset_policy(policy, runner, dones)
        reward_mean = float(torch.as_tensor(rew).float().mean())
        reward_means.append(reward_mean)
        print(
            f"[step {step_id}] reward_mean={reward_mean:.6f}, "
            f"dones={int(torch.as_tensor(dones).sum())}, info_keys={list(extras.keys())[:5]}"
        )
    save_reward_plot(reward_means, args)
    env.close()


def _cli_main() -> None:
    """Parse CLI arguments and launch evaluation. Importing this module is a no-op."""
    args_cli, hydra_args = parse_args()
    sys.argv = [sys.argv[0]] + hydra_args

    @hydra_task_config(args_cli.task, args_cli.policy_agent if args_cli.policy_checkpoint is not None else "")
    def main(env_cfg, agent_cfg) -> None:
        configure_env(env_cfg, args_cli)
        solver_cfg, legacy_use_cuda_graph = build_solver_cfg(args_cli)
        apply_cuda_graph_override(env_cfg, args_cli, legacy_use_cuda_graph=legacy_use_cuda_graph)
        configure_eval_video(env_cfg, args_cli)

        from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

        from isaaclab.app import launch_simulation

        with launch_simulation(build_launch_cfg(env_cfg), args_cli):
            import gymnasium as gym

            gym_kwargs = {"cfg": env_cfg, "device": args_cli.device}
            if solver_cfg is not None:
                gym_kwargs["solver_cfg"] = solver_cfg
            with newton_material_binding_api_autofix():
                env = gym.make(args_cli.task, **gym_kwargs)

            if args_cli.policy_checkpoint is None:
                run_zero_action(env, args_cli)
            else:
                run_policy(env, agent_cfg, args_cli)

    main()  # type: ignore[call-arg]


if __name__ == "__main__":
    _cli_main()
