# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split Allegro reorientation throughput measurements: model-only NeRD inference and policy-driven env steps.

Mirrors :mod:`isaaclab_neural.eval.keyboard_benchmark`. ``model`` times one full-window NeRD model call on
GPU-resident synthetic inputs over a batch sweep. ``env`` runs the teacher closed loop with native MJWarp (CUDA
graph, Newton contacts) or NeRD and times complete control steps, then splits physics from the rest of the step
and, for NeRD, records the solver's ``step_profile`` sections. Run each configuration on an otherwise idle GPU;
results are written as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ALLEGRO_NUM_BODIES = 22
"""Primary contact bodies per world: the Allegro links plus the cube (recorded ``primary_body_q`` width)."""

ALLEGRO_FRAMES_PER_CONTROL_STEP = 4
"""Allegro decimation: 120 Hz physics frames per 30 Hz control step."""


def checkpoint_model_dims(checkpoint: dict[str, Any]) -> tuple[dict[str, int], int]:
    """Return the low-dimensional input widths and the prediction width recorded in a checkpoint."""
    rms = checkpoint["rms_info"]
    input_dims = {name: int(shape[-1]) for name, shape in rms["input_rms"].items()}
    return input_dims, int(rms["output_rms"][-1])


def benchmark_env(args: argparse.Namespace) -> dict[str, Any]:
    """Time policy-driven Allegro control steps with native MJWarp or NeRD dynamics at one world count."""
    import isaaclab_neural.envs  # noqa: F401 - registers the Allegro tasks
    from isaaclab_neural.eval.allegro_reorient_policy import (
        build_allegro_reorient_env_cfg,
        load_allegro_reorient_policy,
        make_allegro_reorient_env,
    )
    from isaaclab_neural.eval.keyboard_benchmark import time_policy_env

    from isaaclab.app import launch_simulation

    cfg, launch_cfg, provenance = build_allegro_reorient_env_cfg(args)
    result: dict[str, Any] = {
        "mode": "env",
        "dynamics": "nerd" if provenance is not None else "native_mjwarp",
        "contacts": "mujoco" if getattr(args, "mujoco_contacts", False) else "newton",
        "nerd": provenance,
        "num_envs": args.num_envs,
        "physics_dt_s": cfg.sim.dt,
        "decimation": cfg.decimation,
        "num_substeps": cfg.sim.physics.num_substeps,
        "gpu": torch.cuda.get_device_name(0),
    }
    try:
        with launch_simulation(launch_cfg, args):
            env = make_allegro_reorient_env(cfg, provenance is not None)
            try:
                wrapped, policy, runner = load_allegro_reorient_policy(
                    env, args.checkpoint, device=cfg.sim.device, seed=args.seed
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
    from isaaclab_neural.eval.allegro_reorient_policy import add_nerd_override_args

    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    model = sub.add_parser("model", help="Model-only NeRD inference over a batch sweep.")
    model.add_argument("--nerd-checkpoint", type=Path, required=True)
    model.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 256, 1024, 2048, 4096, 8192])
    model.add_argument("--valid-tokens", type=int, default=36, help="Valid contact rows per world and frame.")
    model.add_argument("--autocast", choices=("fp32", "fp16", "bf16"), default="fp32")
    model.add_argument("--warmup", type=int, default=10)
    model.add_argument("--iterations", type=int, default=50)
    model.add_argument("--device", default="cuda:0")
    env = sub.add_parser("env", help="Policy-driven end-to-end environment steps at one world count.")
    env.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL teacher checkpoint.")
    add_nerd_override_args(env)
    env.add_argument(
        "--mujoco-contacts",
        action="store_true",
        help="Native MJWarp with the core task's MuJoCo contacts instead of the Newton contacts NeRD is trained on.",
    )
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
        from isaaclab_neural.eval.keyboard_benchmark import benchmark_model
        from isaaclab_neural.utils.checkpoint import load_checkpoint

        checkpoint = load_checkpoint(args.nerd_checkpoint, device="cpu")
        input_dims, prediction_dim = checkpoint_model_dims(checkpoint)
        step_mode = checkpoint["cfg"]["env"]["neural_solver_cfg"].get("step_mode", "physics_frame")
        result = benchmark_model(
            args,
            input_dims=input_dims,
            prediction_dim=prediction_dim,
            num_bodies=ALLEGRO_NUM_BODIES,
            frames_per_control_step=ALLEGRO_FRAMES_PER_CONTROL_STEP,
            frames_per_call=ALLEGRO_FRAMES_PER_CONTROL_STEP if step_mode == "control_step" else 1,
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
