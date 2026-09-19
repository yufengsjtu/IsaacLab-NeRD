# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an RSL-RL control policy in a registered NeRD task without exporting it.

``isaaclab play`` re-exports the actor to TorchScript/ONNX before playing, which the
current RSL-RL release cannot do for multi-group observation actors such as the core
Lift policy. This runner loads the checkpoint the same way dataset generation does and
steps the environment in real time.

Example::

    python -m isaaclab_neural.eval.play_control_policy \\
        --task Isaac-Lift-Cube-Franka-Dataset-Gen-v0 \\
        --checkpoint pretrained/official/Isaac-Lift-Franka_newtonmjwarp_none_rsl_rl.pt \\
        --num_envs 16 --viz newton presets=newton_mjwarp
"""

from __future__ import annotations

import argparse
import sys
import time

import isaaclab_neural.envs  # noqa: F401
import torch
from isaaclab_neural.eval.eval import build_policy_env, load_saved_agent_cfg_for_policy
from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, resolve_task_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Registered task id.")
    parser.add_argument("--checkpoint", required=True, help="RSL-RL checkpoint path.")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-play-mode", action="store_true", help="Keep the training config instead of play_mode().")
    parser.add_argument(
        "--steps", type=int, default=0, help="Stop after this many control steps; 0 runs until the window closes."
    )
    add_launcher_args(parser)
    args, extra = parser.parse_known_args()
    presets = {part.split("=", 1)[1] for part in extra if part.startswith("presets=")}
    presets = {name for value in presets for name in value.split(",") if name} or {"newton_mjwarp"}

    # Hydra re-parses argv inside resolve_task_config; presets are applied explicitly below.
    sys.argv = sys.argv[:1]
    env_cfg, _ = resolve_task_config(args.task, "")
    resolve_presets(env_cfg, presets)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if isinstance(agent_cfg, PresetCfg):
        agent_cfg = agent_cfg.default
    resolve_presets(agent_cfg, presets)
    if not args.no_play_mode and hasattr(env_cfg, "play_mode"):
        env_cfg.play_mode()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    runner_cfg = load_saved_agent_cfg_for_policy(args.checkpoint, agent_cfg, device=args.device, seed=args.seed)

    with launch_simulation(env_cfg, args):
        import gymnasium as gym

        with newton_material_binding_api_autofix():
            env = gym.make(args.task, cfg=env_cfg, device=args.device).unwrapped
        wrapped, policy, _ = build_policy_env(env, runner_cfg, args.checkpoint, args.device)
        obs = wrapped.get_observations()
        step = 0
        try:
            with torch.inference_mode():
                while env.sim.is_headless_or_exist_active_visualizer() and (args.steps == 0 or step < args.steps):
                    start = time.perf_counter()
                    obs, _, _, _ = wrapped.step(policy(obs))
                    step += 1
                    remaining = env.step_dt - (time.perf_counter() - start)
                    if remaining > 0 and env.sim.has_active_visualizers():
                        time.sleep(remaining)
        finally:
            env.close()


if __name__ == "__main__":
    main()
