# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a native SO101 typing policy on the fixed keyboard and optionally record headless Newton GL video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from isaaclab_neural.data.keyboard import KEYBOARD_TASK
from isaaclab_neural.eval.eval import build_policy_env, load_saved_agent_cfg_for_policy, reset_policy
from isaaclab_neural.generate.keyboard_dataset import configure_fixed_keyboard

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

    from isaaclab_tasks.contrib.keyboard.mdp.commands.typing_commands import LetterTypingCommand
    from isaaclab_tasks.contrib.keyboard.so101_env_cfg import SO101KeyboardEnvCfg


def main() -> None:
    import gymnasium as gym

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import resolve_task_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reset-mode", choices=("training", "uniform"), default="uniform")
    parser.add_argument("--reset_ik_iters", type=int, help="Override reset IK iterations for a diagnostic comparison.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--control-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-length", type=int, default=400)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.control_steps <= 0 or args.video_length <= 0:
        parser.error("environment count and rollout/video lengths must be positive")
    if args.reset_ik_iters is not None and args.reset_ik_iters <= 0:
        parser.error("reset IK iterations must be positive")
    if args.report.exists():
        raise FileExistsError(args.report)
    if args.video_dir is not None and args.video_dir.exists():
        raise FileExistsError(args.video_dir)
    sys.argv = [sys.argv[0]]
    cfg, agent = resolve_task_config(KEYBOARD_TASK, "rsl_rl_cfg_entry_point")
    cfg = cast("SO101KeyboardEnvCfg", resolve_presets(cfg, {"newton_mjwarp"}))
    agent = resolve_presets(agent, {"newton_mjwarp"})
    cfg = configure_fixed_keyboard(cfg, variant=0)
    if args.reset_mode == "uniform":
        cfg.commands.typing.reset.enabled = False
    if args.reset_ik_iters is not None:
        cfg.commands.typing.reset.ik_iters = args.reset_ik_iters
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = args.seed
    if args.video_dir is not None:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        camera = cfg.sim.default_visualizer_cfg
        if camera is None:
            raise ValueError("Video recording requires a default visualizer configuration.")
        cfg.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                headless=True,
                eye=camera.eye,
                lookat=camera.lookat,
                window_width=1280,
                window_height=720,
            )
        ]
        cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton_gl",
                output_dir=str(args.video_dir),
                video_length=args.video_length,
                output_filename_prefix="keyboard_policy",
                fps=25,
            )
        ]
        args.visualizer = ["newton_gl"]
        args.visualizer_explicit = True
        args.max_visible_envs = 1
    runner_cfg = load_saved_agent_cfg_for_policy(str(args.checkpoint), agent, device=cfg.sim.device, seed=args.seed)
    episodes = []
    max_force_n = 0.0
    reward_sum = 0.0
    completed_steps = 0
    finite = True
    with launch_simulation(cfg, args):
        env = cast("ManagerBasedRLEnv", gym.make(KEYBOARD_TASK, cfg=cfg).unwrapped)
        wrapped, policy, runner = build_policy_env(env, runner_cfg, str(args.checkpoint), cfg.sim.device)
        with torch.inference_mode():
            env.reset()
            obs = wrapped.get_observations()
            original_reset = env._reset_idx
            command = cast("LetterTypingCommand", env.command_manager.get_term("typing"))
            initial_typed_len = command.typed_len.clone()

            def capture_reset(env_ids):
                command = cast("LetterTypingCommand", env.command_manager.get_term("typing"))
                success = env.termination_manager.get_term("success")[env_ids].clone()
                excessive = env.termination_manager.get_term("excessive_contact")[env_ids].clone()
                abnormal = env.termination_manager.get_term("abnormal_robot")[env_ids].clone()
                for row, world in enumerate(env_ids.tolist()):
                    episodes.append(
                        {
                            "world": world,
                            "success": bool(success[row]),
                            "excessive_contact": bool(excessive[row]),
                            "abnormal_robot": bool(abnormal[row]),
                            "length": int(env.episode_length_buf[world]),
                            "start_distance": int(command._start_distance[world]),
                            "initial_typed_len": int(initial_typed_len[world]),
                            "final_distance": int(command.distance[world]),
                            "target": command.target[world].cpu().tolist(),
                            "typed": command.typed[world].cpu().tolist(),
                        }
                    )
                original_reset(env_ids)
                initial_typed_len[env_ids] = command.typed_len[env_ids]

            env._reset_idx = capture_reset
            try:
                for _ in range(args.control_steps):
                    actions = policy(obs)
                    if not bool(torch.isfinite(actions).all()):
                        finite = False
                        break
                    obs, reward, done, _ = wrapped.step(actions)
                    reset_policy(policy, runner, done)
                    reward_sum += float(reward.sum())
                    completed_steps += 1
                    q = env.scene["robot"].data.joint_pos.torch
                    qd = env.scene["robot"].data.joint_vel.torch
                    forces = env.scene.sensors["robot_contact"].data.net_normal_forces_w_history.torch
                    finite = bool(
                        torch.isfinite(q).all()
                        and torch.isfinite(qd).all()
                        and torch.isfinite(forces).all()
                        and torch.isfinite(reward).all()
                    )
                    max_force_n = max(max_force_n, float(forces.norm(dim=-1).max()))
                    if not finite:
                        break
            finally:
                env._reset_idx = original_reset
                env.close()
    successes = sum(ep["success"] and not ep["excessive_contact"] and not ep["abnormal_robot"] for ep in episodes)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "scene": "fixed",
        "reset_mode": args.reset_mode,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "requested_control_steps": args.control_steps,
        "reset_ik_iters": cfg.commands.typing.reset.ik_iters,
        "finite": finite,
        "completed_control_steps": completed_steps,
        "completed_episodes": len(episodes),
        "clean_successes": successes,
        "clean_success_rate": successes / len(episodes) if episodes else None,
        "episodes_with_empty_initial_buffer": sum(ep["initial_typed_len"] == 0 for ep in episodes),
        "excessive_contact_episodes": sum(ep["excessive_contact"] for ep in episodes),
        "abnormal_robot_episodes": sum(ep["abnormal_robot"] for ep in episodes),
        "reward_sum": reward_sum,
        "max_observed_force_n_after_step": max_force_n,
        "video_dir": str(args.video_dir) if args.video_dir else None,
        "policy_accepted": False,
        "requires_video_and_multiseed_review": True,
        "episodes": episodes,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not finite:
        raise RuntimeError("Policy evaluation produced non-finite values.")


if __name__ == "__main__":
    main()
