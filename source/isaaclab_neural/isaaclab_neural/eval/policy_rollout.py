# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL policy closed loop in a manager-based task under native MJWarp or NeRD dynamics.

The same task metrics are computed for both backends, so a native report can serve as the fidelity reference that
selects NeRD checkpoints during training (``algorithm.eval.mode: policy_rollout``). Training configures metrics as
``{"name": ..., ...}`` entries; the command line selects them by name with their default parameters:

* ``velocity_tracking_error``: mean planar speed error [m/s] between the ``command`` velocity and the robot's base
  linear velocity.
* ``pose_success_rate``: fraction of world steps whose ``command`` pose term reports a position error below
  ``position_threshold`` [m] (and, when tracked, an orientation error below ``orientation_threshold`` [rad]).

Every report also contains completed episodes, the failure rate (episodes ended by a termination other than a
timeout), the mean episode length and the mean reward per world step. ``--video-dir`` records world 0 with the
headless Newton GL visualizer; ``--follow-asset`` keeps the camera on a moving robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from isaaclab_neural.eval.eval import build_policy_env, load_saved_agent_cfg_for_policy, reset_policy

METRIC_NAMES = ("velocity_tracking_error", "pose_success_rate")
"""Task metrics supported by :func:`rollout_task_policy`."""


def load_task_policy(env: Any, checkpoint: Path, *, task: str, device: str, seed: int):
    """Wrap ``env`` and load the RSL-RL policy, preferring the agent config saved beside the checkpoint.

    Args:
        env: The unwrapped manager-based environment.
        checkpoint: RSL-RL policy checkpoint.
        task: Registered task whose ``rsl_rl_cfg_entry_point`` supplies the fallback runner config.
        device: Policy device.
        seed: Runner seed.

    Returns:
        The wrapped environment, the inference policy and its runner.
    """
    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import resolve_task_config

    # Explicit empty overrides keep the caller's command line (e.g. training arguments) out of Hydra.
    _, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    agent = resolve_presets(agent, {"newton_mjwarp"})
    runner_cfg = load_saved_agent_cfg_for_policy(str(checkpoint), agent, device=device, seed=seed)
    # During DDP NeRD training the policy runs on rank 0 only. RSL-RL infers distributed mode from
    # WORLD_SIZE and would initialize the already-initialized default process group a second time.
    world_size = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = "1"
    try:
        return build_policy_env(env, runner_cfg, str(checkpoint), device)
    finally:
        if world_size is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = world_size


def _metric_values(env: Any, metric: dict[str, Any]) -> torch.Tensor:
    """Return one per-world value of ``metric`` for the current step."""
    name = metric["name"]
    if name == "velocity_tracking_error":
        command = env.command_manager.get_command(metric.get("command", "base_velocity"))[:, :2]
        velocity = env.scene[metric.get("asset", "robot")].data.root_lin_vel_b.torch[:, :2]
        return torch.linalg.norm(command - velocity, dim=-1)
    if name == "pose_success_rate":
        errors = env.command_manager.get_term(metric.get("command", "object_pose")).metrics
        success = errors["position_error"] < float(metric.get("position_threshold", 0.05))
        if "orientation_error" in errors:
            success &= errors["orientation_error"] < float(metric.get("orientation_threshold", 0.5))
        return success.float()
    raise ValueError(f"Unknown rollout metric {name!r}; expected one of {METRIC_NAMES}.")


def rollout_task_policy(
    env: Any,
    wrapped: Any,
    policy: Any,
    runner: Any,
    control_steps: int,
    metrics: list[dict[str, Any]] | tuple = (),
    on_step: Any = None,
) -> dict[str, Any]:
    """Reset ``env`` and run the policy closed loop, recording every episode that ends.

    Call under :func:`torch.inference_mode`. Episodes still running after ``control_steps`` are not recorded.
    ``on_step(env)`` is called after every control step.

    Returns:
        Completed episodes, failure rate, mean episode length, mean reward per world step, the configured task metrics
        (averaged over worlds and steps), whether states and rewards stayed finite and the completed control steps.
    """
    for metric in metrics:
        if metric.get("name") not in METRIC_NAMES:
            raise ValueError(f"Unknown rollout metric {metric.get('name')!r}; expected one of {METRIC_NAMES}.")
    episodes: list[dict[str, Any]] = []
    env.reset()
    obs = wrapped.get_observations()
    original_reset = env._reset_idx
    terminations = env.termination_manager

    def capture_reset(env_ids):
        failed = terminations.terminated[env_ids].clone()
        for row, world in enumerate(env_ids.tolist()):
            episodes.append({"world": world, "failed": bool(failed[row]), "length": int(env.episode_length_buf[world])})
        original_reset(env_ids)

    sums = {metric["name"]: 0.0 for metric in metrics}
    finite, completed, reward_sum = True, 0, 0.0
    env._reset_idx = capture_reset
    try:
        for _ in range(control_steps):
            actions = policy(obs)
            if not bool(torch.isfinite(actions).all()):
                finite = False
                break
            obs, reward, done, _ = wrapped.step(actions)
            reset_policy(policy, runner, done)
            reward_sum += float(reward.mean())
            for metric in metrics:
                sums[metric["name"]] += float(_metric_values(env, metric).mean())
            completed += 1
            if on_step is not None:
                on_step(env)
            finite = bool(
                torch.isfinite(env.scene["robot"].data.joint_pos.torch).all() and torch.isfinite(reward).all()
            )
            if not finite:
                break
    finally:
        env._reset_idx = original_reset
    count = len(episodes)
    failures = sum(episode["failed"] for episode in episodes)
    return {
        "episodes": count,
        "failed_episodes": failures,
        "failure_rate": failures / count if count else None,
        "mean_episode_length": sum(episode["length"] for episode in episodes) / count if count else None,
        "reward_per_step": reward_sum / completed if completed else None,
        **{name: value / completed if completed else None for name, value in sums.items()},
        "finite": finite,
        "completed_control_steps": completed,
        "episode_records": episodes,
    }


def selection_error(summary: dict[str, Any], selection: dict[str, Any]) -> float:
    """Return the checkpoint-selection error of a rollout summary (lower is better, ``inf`` when invalid).

    With a ``reference`` value the error is the absolute distance of ``metric`` to it, so the checkpoint closest to the
    native dynamics is kept instead of the most optimistic one. Otherwise ``higher_is_better`` orients the metric.
    """
    value = summary.get(selection["metric"])
    if not summary.get("finite") or value is None:
        return float("inf")
    if selection.get("reference") is not None:
        return abs(float(value) - float(selection["reference"]))
    return -float(value) if selection.get("higher_is_better", False) else float(value)


def add_nerd_override_args(parser: argparse.ArgumentParser) -> None:
    """Add the NeRD checkpoint and online-inference override options."""
    parser.add_argument("--nerd-checkpoint", type=Path, help="Simulate with this NeRD checkpoint instead of MJWarp.")
    parser.add_argument("--nerd-task", help="Registered NeRD task used with --nerd-checkpoint.")
    parser.add_argument("--frame-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--inference-precision", choices=("fp32", "tf32", "fp16", "bf16"))
    parser.add_argument("--inference-compile", action=argparse.BooleanOptionalAction, default=None)


def build_task_env(args: argparse.Namespace):
    """Resolve the native task, or the NeRD task with the checkpoint's solver, for ``args.num_envs`` worlds.

    Returns:
        The environment config, the config to launch the simulation with, the task to create and the NeRD solver
        provenance (``None`` for native MJWarp).
    """
    from isaaclab_neural.eval.eval import build_launch_cfg
    from isaaclab_neural.physics import NerdSolverCfg
    from isaaclab_neural.utils.checkpoint import load_checkpoint

    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import resolve_task_config

    nerd = args.nerd_checkpoint is not None
    if nerd and not args.nerd_task:
        raise ValueError("--nerd-checkpoint requires --nerd-task.")
    task = args.nerd_task if nerd else args.task
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    cfg = resolve_presets(cfg, {"newton_mjwarp"})
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = args.seed
    if not nerd:
        return cfg, cfg, task, None
    checkpoint = load_checkpoint(args.nerd_checkpoint, device="cpu")
    solver_cfg = NerdSolverCfg(**checkpoint["cfg"]["env"]["neural_solver_cfg"])
    for name, value in (
        ("use_frame_cache", args.frame_cache),
        ("inference_precision", args.inference_precision),
        ("inference_compile", args.inference_compile),
    ):
        if value is not None:
            setattr(solver_cfg, name, value)
    solver_cfg.neural_model_path = str(args.nerd_checkpoint.resolve())
    cfg.sim.physics.solver_cfg = solver_cfg
    with args.nerd_checkpoint.open("rb") as stream:
        provenance = {
            "checkpoint": str(args.nerd_checkpoint.resolve()),
            "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
            "epoch": checkpoint.get("epoch"),
            "step_mode": solver_cfg.step_mode,
            "use_frame_cache": solver_cfg.use_frame_cache,
            "inference_precision": solver_cfg.inference_precision,
            "inference_compile": solver_cfg.inference_compile,
        }
    return cfg, build_launch_cfg(cfg), task, provenance


def add_video_args(parser: argparse.ArgumentParser) -> None:
    """Add headless Newton GL video options."""
    parser.add_argument("--video-dir", type=Path, help="Record world 0 to this new directory.")
    parser.add_argument("--video-length", type=int, default=500, help="Recorded control steps.")
    parser.add_argument("--camera-eye", type=float, nargs=3, help="Camera position [m] (offset when following).")
    parser.add_argument("--camera-lookat", type=float, nargs=3, help="Camera target [m]; defaults to the task view.")
    parser.add_argument("--follow-asset", help="Keep the camera on this asset's root in world 0.")


def configure_video(cfg: Any, launch_cfg: Any, args: argparse.Namespace) -> None:
    """Record world 0 at the control rate with the headless Newton GL visualizer (also on the launch config)."""
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    camera = cfg.sim.default_visualizer_cfg
    fps = round(1.0 / (cfg.sim.dt * cfg.decimation))
    for target in {id(cfg): cfg, id(launch_cfg): launch_cfg}.values():
        target.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                headless=True,
                eye=tuple(args.camera_eye) if args.camera_eye else camera.eye,
                lookat=tuple(args.camera_lookat) if args.camera_lookat else camera.lookat,
                window_width=1280,
                window_height=720,
            )
        ]
        target.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton_gl",
                output_dir=str(args.video_dir),
                video_length=args.video_length,
                output_filename_prefix="policy_rollout",
                fps=fps,
            )
        ]
    args.visualizer = ["newton_gl"]
    args.visualizer_explicit = True
    args.max_visible_envs = 1


def follow_camera(asset: str, offset: tuple[float, float, float]):
    """Return an ``on_step`` callback that places the Newton GL camera at ``offset`` from the asset root."""

    def update(env: Any) -> None:
        position = env.scene[asset].data.root_pos_w.torch[0].tolist()
        for visualizer in getattr(env.sim, "_visualizers", []):
            if hasattr(visualizer, "set_camera_view"):
                eye = [value + delta for value, delta in zip(position, offset)]
                visualizer.set_camera_view(eye, [position[0], position[1], position[2] - 0.1])

    return update


def main() -> None:
    import gymnasium as gym
    import isaaclab_neural.envs  # noqa: F401 - registers the NeRD tasks
    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    from isaaclab.app import add_launcher_args, launch_simulation

    import isaaclab_tasks  # noqa: F401 - registers the core and contrib tasks

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Native MJWarp task.")
    parser.add_argument("--policy-task", help="Task supplying the fallback RSL-RL runner config; defaults to --task.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL policy checkpoint.")
    add_nerd_override_args(parser)
    parser.add_argument(
        "--metric", action="append", default=[], choices=METRIC_NAMES, help="Task metric with default parameters."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--control-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100)
    add_video_args(parser)
    add_launcher_args(parser)
    args = parser.parse_args()
    metrics = [{"name": name} for name in args.metric]
    if args.num_envs <= 0 or args.control_steps <= 0 or args.video_length <= 0:
        parser.error("environment count and rollout/video lengths must be positive")
    if args.report.exists():
        raise FileExistsError(args.report)
    if args.video_dir is not None and args.video_dir.exists():
        raise FileExistsError(args.video_dir)
    sys.argv = [sys.argv[0]]
    cfg, launch_cfg, task, provenance = build_task_env(args)
    on_step = None
    if args.video_dir is not None:
        configure_video(cfg, launch_cfg, args)
        if args.follow_asset:
            on_step = follow_camera(args.follow_asset, tuple(args.camera_eye or (2.5, 2.5, 1.5)))
    with launch_simulation(launch_cfg, args):
        with newton_material_binding_api_autofix():
            env = gym.make(task, cfg=cfg).unwrapped
        try:
            wrapped, policy, runner = load_task_policy(
                env, args.checkpoint, task=args.policy_task or args.task, device=cfg.sim.device, seed=args.seed
            )
            with torch.inference_mode():
                rollout = rollout_task_policy(env, wrapped, policy, runner, args.control_steps, metrics, on_step)
        finally:
            env.close()
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "task": task,
        "dynamics": "native_mjwarp" if provenance is None else "nerd",
        "nerd": provenance,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "requested_control_steps": args.control_steps,
        "metrics": metrics,
        **rollout,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "episode_records"}, indent=2))
    if not rollout["finite"]:
        raise RuntimeError("Policy evaluation produced non-finite values.")


if __name__ == "__main__":
    main()
