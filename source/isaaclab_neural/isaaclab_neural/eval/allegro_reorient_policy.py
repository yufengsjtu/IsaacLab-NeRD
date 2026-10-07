# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an Allegro cube-reorientation policy closed loop under MJWarp or NeRD dynamics.

By default Newton MJWarp (with Newton contacts, as used for data collection) simulates the
hand and the cube. ``--nerd-checkpoint`` replaces it with learned NeRD dynamics while the
task, policy, resets and termination thresholds stay unchanged, so goals per episode and
drop rates are directly comparable. ``--video-dir`` records world 0 with the headless Newton GL
visualizer and writes a per-control-step ``reorient_overlay.json`` (goals, episodes, drops).
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

ALLEGRO_REORIENT_POLICY_TASK = "Isaac-Reorient-Cube-Allegro-Policy-v0"
"""Task whose dynamics, observations and resets the policy and the NeRD data share."""

ALLEGRO_REORIENT_CORE_TASK = "Isaac-Reorient-Cube-Allegro"
"""Core task with MuJoCo's internal contacts, an alternative native-MJWarp throughput baseline."""


def summarize_reorient_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate completed episodes into goals per episode and drop rate.

    Args:
        episodes: Records with integer ``goals`` and boolean ``dropped`` entries.

    Returns:
        Episode count, mean goals per episode, drop count and drop rate (``None`` without episodes).
    """
    count = len(episodes)
    drops = sum(episode["dropped"] for episode in episodes)
    return {
        "episodes": count,
        "mean_goals_per_episode": sum(episode["goals"] for episode in episodes) / count if count else None,
        "drops": drops,
        "drop_rate": drops / count if count else None,
    }


def load_allegro_reorient_policy(env: Any, checkpoint: Path, *, device: str, seed: int):
    """Wrap ``env`` and load the RSL-RL policy, preferring the agent config saved beside the checkpoint.

    Returns:
        The wrapped environment, the inference policy and its runner.
    """
    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import resolve_task_config

    # Explicit empty overrides keep the caller's command line (e.g. training arguments) out of Hydra.
    _, agent = resolve_task_config(ALLEGRO_REORIENT_POLICY_TASK, "rsl_rl_cfg_entry_point", overrides=[])
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


def rollout_reorient_policy(
    env: Any, wrapped: Any, policy: Any, runner: Any, control_steps: int, on_step: Any = None
) -> dict[str, Any]:
    """Reset ``env`` and run the policy closed loop, recording every episode that ends.

    Call under :func:`torch.inference_mode`. Episodes still running after ``control_steps`` are not recorded.
    ``on_step(env, episodes)`` is called after every control step.

    Returns:
        Recorded episodes, whether states and rewards stayed finite, completed control steps and the reward sum.
    """
    episodes: list[dict[str, Any]] = []
    env.reset()
    obs = wrapped.get_observations()
    original_reset = env._reset_idx
    command = env.command_manager.get_term("object_pose")

    def capture_reset(env_ids):
        goals = command._success.snapshot(env_ids).clone()
        dropped = env.termination_manager.get_term("object_out_of_reach")[env_ids].clone()
        for row, world in enumerate(env_ids.tolist()):
            episodes.append(
                {
                    "world": world,
                    "goals": int(goals[row]),
                    "dropped": bool(dropped[row]),
                    "length": int(env.episode_length_buf[world]),
                }
            )
        original_reset(env_ids)

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
            reward_sum += float(reward.sum())
            completed += 1
            if on_step is not None:
                on_step(env, episodes)
            finite = bool(
                torch.isfinite(env.scene["robot"].data.joint_pos.torch).all()
                and torch.isfinite(env.scene["object"].data.root_pose_w.torch).all()
                and torch.isfinite(reward).all()
            )
            if not finite:
                break
    finally:
        env._reset_idx = original_reset
    return {"episodes": episodes, "finite": finite, "completed_control_steps": completed, "reward_sum": reward_sum}


def add_nerd_override_args(parser: argparse.ArgumentParser) -> None:
    """Add the NeRD checkpoint and online-inference override options."""
    parser.add_argument("--nerd-checkpoint", type=Path, help="Simulate with this NeRD checkpoint instead of MJWarp.")
    parser.add_argument(
        "--frame-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint's NeRD use_frame_cache.",
    )
    parser.add_argument(
        "--inference-precision",
        choices=("fp32", "tf32", "fp16", "bf16"),
        help="Override the checkpoint's NeRD online inference precision.",
    )
    parser.add_argument(
        "--inference-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the NeRD inference_compile option (torch.compile of fixed-shape steady-state inference steps).",
    )


def build_allegro_reorient_env_cfg(args: argparse.Namespace):
    """Resolve the policy task for ``args.num_envs`` worlds, swapping in NeRD when ``args.nerd_checkpoint`` is set.

    ``args.mujoco_contacts`` (optional) selects the core task's native MuJoCo contacts instead, for MJWarp only.

    Returns:
        The environment config, the config to launch the simulation with and the NeRD provenance (``None`` for
        native MJWarp).
    """
    import copy

    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import resolve_task_config

    mujoco_contacts = bool(getattr(args, "mujoco_contacts", False))
    if mujoco_contacts and args.nerd_checkpoint is not None:
        raise ValueError("MuJoCo contacts are a native-MJWarp baseline; NeRD uses Newton contacts.")
    task = ALLEGRO_REORIENT_CORE_TASK if mujoco_contacts else ALLEGRO_REORIENT_POLICY_TASK
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    cfg = resolve_presets(cfg, {"newton_mjwarp"})
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = args.seed
    if args.nerd_checkpoint is None:
        return cfg, cfg, None
    from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
    from isaaclab_neural.utils.checkpoint import load_checkpoint

    nerd = load_checkpoint(args.nerd_checkpoint, device="cpu")
    launch_cfg = copy.deepcopy(cfg)
    launch_cfg.sim.physics.use_cuda_graph = False
    solver_cfg = NerdSolverCfg(**nerd["cfg"]["env"]["neural_solver_cfg"])
    for name, value in (
        ("use_frame_cache", args.frame_cache),
        ("inference_precision", args.inference_precision),
        ("inference_compile", args.inference_compile),
    ):
        if value is not None:
            setattr(solver_cfg, name, value)
    solver_cfg.neural_model_path = str(args.nerd_checkpoint.resolve())
    teacher = cfg.sim.physics
    cfg.sim.physics = NerdNewtonCfg(
        solver_cfg=solver_cfg,
        collision_cfg=teacher.collision_cfg,
        default_shape_cfg=teacher.default_shape_cfg,
        num_substeps=1,
        use_cuda_graph=False,
    )
    with args.nerd_checkpoint.open("rb") as stream:
        provenance = {
            "checkpoint": str(args.nerd_checkpoint.resolve()),
            "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
            "epoch": nerd.get("epoch"),
            "use_frame_cache": solver_cfg.use_frame_cache,
            "inference_precision": solver_cfg.inference_precision,
            "inference_compile": solver_cfg.inference_compile,
        }
    return cfg, launch_cfg, provenance


def add_video_args(parser: argparse.ArgumentParser) -> None:
    """Add headless Newton GL video options."""
    parser.add_argument("--video-dir", type=Path, help="Record world 0 to this new directory.")
    parser.add_argument("--video-length", type=int, default=900, help="Recorded control steps.")
    parser.add_argument(
        "--camera-eye", type=float, nargs=3, help="Video camera position [m]; defaults to the task view."
    )
    parser.add_argument(
        "--camera-lookat", type=float, nargs=3, help="Video camera target [m]; defaults to the task view."
    )


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
                output_filename_prefix="allegro_reorient",
                fps=fps,
            )
        ]
    args.visualizer = ["newton_gl"]
    args.visualizer_explicit = True
    args.max_visible_envs = 1


def make_allegro_reorient_env(cfg, nerd: bool):
    """Create the native MJWarp policy environment or its NeRD counterpart from a resolved config."""
    if not nerd:
        import gymnasium as gym

        return gym.make(ALLEGRO_REORIENT_POLICY_TASK, cfg=cfg).unwrapped
    from isaaclab_neural.envs.allegro_reorient_nerd_env import NerdAllegroReorientEnv

    return NerdAllegroReorientEnv(cfg=cfg)


def main() -> None:
    import isaaclab_neural.envs  # noqa: F401 - registers the Allegro NeRD tasks

    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL policy checkpoint.")
    add_nerd_override_args(parser)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--control-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=100)
    add_video_args(parser)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.control_steps <= 0 or args.video_length <= 0:
        parser.error("environment count and rollout/video lengths must be positive")
    if args.report.exists():
        raise FileExistsError(args.report)
    if args.video_dir is not None and args.video_dir.exists():
        raise FileExistsError(args.video_dir)
    sys.argv = [sys.argv[0]]
    cfg, launch_cfg, nerd_provenance = build_allegro_reorient_env_cfg(args)
    if args.video_dir is not None:
        configure_video(cfg, launch_cfg, args)
    overlay: list[dict[str, Any]] = []

    def record_overlay(env, episodes):
        # Headless capture does not dispatch marker callbacks; refresh the goal marker for the next frame.
        env.sim.vis_marker_registry.dispatch_callbacks()
        ended = [episode for episode in episodes if episode["world"] == 0]
        overlay.append(
            {
                "goals": int(env.command_manager.get_term("object_pose").metrics["consecutive_success"][0]),
                "episodes": len(ended),
                "drops": sum(episode["dropped"] for episode in ended),
            }
        )

    with launch_simulation(launch_cfg, args):
        env = make_allegro_reorient_env(cfg, nerd_provenance is not None)
        try:
            wrapped, policy, runner = load_allegro_reorient_policy(
                env, args.checkpoint, device=cfg.sim.device, seed=args.seed
            )
            with torch.inference_mode():
                rollout = rollout_reorient_policy(
                    env,
                    wrapped,
                    policy,
                    runner,
                    args.control_steps,
                    on_step=record_overlay if args.video_dir is not None else None,
                )
        finally:
            env.close()
    episodes = rollout["episodes"]
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dynamics": "native_mjwarp" if nerd_provenance is None else "nerd",
        "nerd": nerd_provenance,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "requested_control_steps": args.control_steps,
        "finite": rollout["finite"],
        "completed_control_steps": rollout["completed_control_steps"],
        **summarize_reorient_episodes(episodes),
        "reward_sum": rollout["reward_sum"],
        "episode_records": episodes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.video_dir is not None:
        args.video_dir.mkdir(parents=True, exist_ok=True)
        (args.video_dir / "reorient_overlay.json").write_text(json.dumps({"frames": overlay}) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "episode_records"}, indent=2))
    if not rollout["finite"]:
        raise RuntimeError("Policy evaluation produced non-finite values.")


if __name__ == "__main__":
    main()
