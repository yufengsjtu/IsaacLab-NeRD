# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split throughput measurements for any manager-based NeRD task: model-only inference and policy-driven env steps.

Generalizes :mod:`isaaclab_neural.eval.allegro_benchmark`. ``model`` times one full-window NeRD model call on
GPU-resident synthetic inputs over a batch sweep (``--num-bodies`` primary contact bodies, ``--frames-per-control-step``
from the task's decimation). ``env`` runs the teacher closed loop in ``--task`` with native MJWarp or, with
``--nerd-checkpoint``, in ``--nerd-task`` with NeRD, and times complete control steps. Run each configuration on an
otherwise idle GPU; results are written as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def benchmark_env(args: argparse.Namespace) -> dict[str, Any]:
    """Time policy-driven control steps with native MJWarp or NeRD dynamics at one world count."""
    import gymnasium as gym
    import isaaclab_neural.envs  # noqa: F401 - registers the NeRD tasks
    from isaaclab_neural.eval.keyboard_benchmark import time_policy_env
    from isaaclab_neural.eval.policy_rollout import build_task_env, load_task_policy
    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    from isaaclab.app import launch_simulation

    import isaaclab_tasks  # noqa: F401 - registers the core and contrib tasks

    cfg, launch_cfg, task, provenance = build_task_env(args)
    result: dict[str, Any] = {
        "mode": "env",
        "task": task,
        "dynamics": "nerd" if provenance is not None else "native_mjwarp",
        "nerd": provenance,
        "num_envs": args.num_envs,
        "physics_dt_s": cfg.sim.dt,
        "decimation": cfg.decimation,
        "gpu": torch.cuda.get_device_name(0),
    }
    try:
        with launch_simulation(launch_cfg, args):
            with newton_material_binding_api_autofix():
                env = gym.make(task, cfg=cfg).unwrapped
            try:
                wrapped, policy, runner = load_task_policy(
                    env, args.checkpoint, task=args.policy_task or args.task, device=cfg.sim.device, seed=args.seed
                )
                result.update(time_policy_env(env, wrapped, policy, runner, args))
                from isaaclab_newton.physics.newton_manager import NewtonManager

                result["cuda_graph_active"] = NewtonManager._graph is not None
            finally:
                env.close()
    except torch.OutOfMemoryError as error:
        result.update(status="oom", error=str(error)[:300])
    return result


def main() -> None:
    from isaaclab_neural.eval.policy_rollout import add_nerd_override_args

    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    model = sub.add_parser("model", help="Model-only NeRD inference over a batch sweep.")
    model.add_argument("--nerd-checkpoint", type=Path, required=True)
    model.add_argument("--num-bodies", type=int, required=True, help="Primary contact bodies per world.")
    model.add_argument("--frames-per-control-step", type=int, required=True, help="Task decimation.")
    model.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 256, 1024, 2048, 4096, 8192])
    model.add_argument("--valid-tokens", type=int, default=16, help="Valid contact rows per world and frame.")
    model.add_argument("--autocast", choices=("fp32", "fp16", "bf16"), default="fp32")
    model.add_argument("--warmup", type=int, default=10)
    model.add_argument("--iterations", type=int, default=50)
    model.add_argument("--device", default="cuda:0")
    env = sub.add_parser("env", help="Policy-driven end-to-end environment steps at one world count.")
    env.add_argument("--task", required=True, help="Native MJWarp task.")
    env.add_argument("--policy-task", help="Task supplying the fallback RSL-RL runner config; defaults to --task.")
    env.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL teacher checkpoint.")
    add_nerd_override_args(env)
    env.add_argument("--num-envs", type=int, required=True)
    env.add_argument("--seed", type=int, default=43)
    env.add_argument("--warmup", type=int, default=30)
    env.add_argument("--steps", type=int, default=200)
    env.add_argument("--breakdown-steps", type=int, default=40)
    env.add_argument("--profile-steps", type=int, default=20, help="NeRD step_profile control steps; 0 disables.")
    add_launcher_args(env)
    for sub_parser in (model, env):
        sub_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    sys.argv = [sys.argv[0]]
    if args.mode == "model":
        from isaaclab_neural.eval.allegro_benchmark import checkpoint_model_dims
        from isaaclab_neural.eval.keyboard_benchmark import benchmark_model
        from isaaclab_neural.utils.checkpoint import load_checkpoint

        checkpoint = load_checkpoint(args.nerd_checkpoint, device="cpu")
        input_dims, prediction_dim = checkpoint_model_dims(checkpoint)
        step_mode = checkpoint["cfg"]["env"]["neural_solver_cfg"].get("step_mode", "physics_frame")
        result = benchmark_model(
            args,
            input_dims=input_dims,
            prediction_dim=prediction_dim,
            num_bodies=args.num_bodies,
            frames_per_control_step=args.frames_per_control_step,
            frames_per_call=args.frames_per_control_step if step_mode == "control_step" else 1,
        )
        result["step_mode"] = step_mode
    else:
        args.device = args.device or "cuda:0"
        result = benchmark_env(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print("BENCH", json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
