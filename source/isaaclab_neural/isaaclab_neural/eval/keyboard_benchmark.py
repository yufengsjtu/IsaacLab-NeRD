# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split SO101 Keyboard throughput measurements: model-only NeRD inference and policy-driven environment steps.

``model`` times one NeRD model call on GPU-resident synthetic inputs (no simulator, collision, contact packing
or state conversion) with CUDA events over a batch sweep. ``env`` runs the typing policy closed loop with native
MJWarp or NeRD dynamics and times complete environment steps; a second phase synchronizes around each physics
frame to split physics from the rest of the step, and for NeRD a third phase records the solver's
``step_profile`` sections. Warm-up steps are excluded. Results are written as JSON, one configuration per run.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

PHYSICS_FRAMES_PER_CONTROL_STEP = 4
"""SO101 Keyboard decimation: 100 Hz physics frames per 25 Hz control step."""


def synthetic_model_inputs(cfg: dict[str, Any], batch: int, valid_tokens: int, device: str) -> dict[str, torch.Tensor]:
    """Build normalized-scale model inputs with ``valid_tokens`` owner-frame contact rows per world."""
    from isaaclab_neural.contacts.contact_set_schema import (
        CONTACT_TOKEN_BODY_SLOT_INDEX,
        CONTACT_TOKEN_OTHER_BODY_SLOT_INDEX,
        CONTACT_TOKEN_VALID_INDEX,
    )

    solver = cfg["env"]["neural_solver_cfg"]
    history, capacity = int(solver["num_states_history"]), int(solver["max_contact_tokens"])
    if not 0 <= valid_tokens <= capacity:
        raise ValueError(f"valid_tokens must lie in [0, {capacity}].")
    generator = torch.Generator(device=device).manual_seed(0)
    shape = (batch, history)
    tokens = torch.randn((*shape, capacity, 17), generator=generator, device=device)
    tokens[..., CONTACT_TOKEN_VALID_INDEX] = (torch.arange(capacity, device=device) < valid_tokens).float()
    tokens[..., CONTACT_TOKEN_BODY_SLOT_INDEX] = torch.randint(
        0, 116, (*shape, capacity), generator=generator, device=device
    ).float()
    tokens[..., CONTACT_TOKEN_OTHER_BODY_SLOT_INDEX] = -1.0
    return {
        "states_embedding": torch.randn((*shape, 228), generator=generator, device=device),
        "joint_f": torch.randn((*shape, 114), generator=generator, device=device),
        "gravity_dir": torch.randn((*shape, 3), generator=generator, device=device),
        "contact_tokens": tokens,
    }


def benchmark_model(args: argparse.Namespace) -> dict[str, Any]:
    """Time full-window NeRD model calls exactly as the solver issues them (``return_aux``, eval mode)."""
    from isaaclab_neural.utils.checkpoint import load_checkpoint, reconstruct_model_from_checkpoint

    checkpoint = load_checkpoint(args.nerd_checkpoint, device="cpu")
    cfg = checkpoint["cfg"]
    sample = synthetic_model_inputs(cfg, 1, args.valid_tokens, args.device)
    solver = SimpleNamespace(
        get_neural_model_inputs=lambda: sample,
        prediction_dim=228,
        contact_mode="newton_native",
        num_contact_bodies_per_env=116,
    )
    model = reconstruct_model_from_checkpoint(checkpoint, solver, device=args.device)
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.autocast]
    rows = []
    for batch in args.batch_sizes:
        inputs = synthetic_model_inputs(cfg, batch, args.valid_tokens, args.device)
        try:
            with (
                torch.inference_mode(),
                torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None),
            ):
                for _ in range(args.warmup):
                    model(dict(inputs), return_aux=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    model(dict(inputs), return_aux=True)
                end.record()
                torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            rows.append({"batch": batch, "status": "oom"})
            torch.cuda.empty_cache()
            continue
        ms = start.elapsed_time(end) / args.iterations
        rows.append(
            {
                "batch": batch,
                "status": "ok",
                "model_call_ms": ms,
                "model_calls_per_s": 1000.0 / ms,
                "physics_frames_per_s": batch * 1000.0 / ms,
                "control_steps_per_s_model_only": batch * 1000.0 / (ms * PHYSICS_FRAMES_PER_CONTROL_STEP),
                "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
        print("MODEL", json.dumps(rows[-1]), flush=True)
    return {
        "mode": "model",
        "scope": "One full-window model call on GPU-resident synthetic inputs; excludes simulator, collision, "
        "contact packing, input preparation and state conversion.",
        "nerd_checkpoint": str(args.nerd_checkpoint),
        "epoch": checkpoint.get("epoch"),
        "autocast": args.autocast,
        "valid_contact_tokens_per_frame": args.valid_tokens,
        "history": cfg["env"]["neural_solver_cfg"]["num_states_history"],
        "gpu": torch.cuda.get_device_name(0),
        "rows": rows,
    }


def benchmark_env(args: argparse.Namespace, launcher_args: argparse.Namespace) -> dict[str, Any]:
    """Time policy-driven environment steps with native MJWarp or NeRD dynamics at one world count."""
    import gymnasium as gym
    from isaaclab_neural.data.keyboard import KEYBOARD_TASK
    from isaaclab_neural.eval.keyboard_policy import load_keyboard_policy
    from isaaclab_neural.generate.keyboard_dataset import configure_fixed_keyboard

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import resolve_task_config

    cfg, _ = resolve_task_config(KEYBOARD_TASK, "rsl_rl_cfg_entry_point", overrides=[])
    cfg = configure_fixed_keyboard(resolve_presets(cfg, {"newton_mjwarp"}), variant=0)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    if args.robot_usd is not None:
        cfg.scene.robot.spawn.usd_path = str(args.robot_usd.resolve())
    launch_cfg = cfg
    if args.nerd_checkpoint is not None:
        from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
        from isaaclab_neural.utils.checkpoint import load_checkpoint

        nerd = load_checkpoint(args.nerd_checkpoint, device="cpu")
        launch_cfg = copy.deepcopy(cfg)
        launch_cfg.sim.physics.use_cuda_graph = False
        solver_cfg = NerdSolverCfg(**nerd["cfg"]["env"]["neural_solver_cfg"])
        if args.frame_cache is not None:
            solver_cfg.use_frame_cache = args.frame_cache
        solver_cfg.neural_model_path = str(args.nerd_checkpoint.resolve())
        teacher = cfg.sim.physics
        cfg.sim.physics = NerdNewtonCfg(
            solver_cfg=solver_cfg,
            collision_cfg=teacher.collision_cfg,
            default_shape_cfg=teacher.default_shape_cfg,
            num_substeps=1,
            use_cuda_graph=False,
        )
        cfg.robot_normal_force_sensor = "robot_contact"
    result: dict[str, Any] = {
        "mode": "env",
        "dynamics": "nerd" if args.nerd_checkpoint is not None else "native_mjwarp",
        "nerd_checkpoint": str(args.nerd_checkpoint) if args.nerd_checkpoint else None,
        "num_envs": args.num_envs,
        "physics_dt_s": cfg.sim.dt,
        "decimation": cfg.decimation,
        "num_substeps": cfg.sim.physics.num_substeps,
        "use_frame_cache": getattr(getattr(cfg.sim.physics, "solver_cfg", None), "use_frame_cache", None),
        "gpu": torch.cuda.get_device_name(0),
    }
    try:
        with launch_simulation(launch_cfg, launcher_args):
            if args.nerd_checkpoint is None:
                env = gym.make(KEYBOARD_TASK, cfg=cfg).unwrapped
            else:
                from isaaclab_neural.envs.keyboard_nerd_env import NerdSO101KeyboardEnv

                env = NerdSO101KeyboardEnv(cfg=cfg)
            try:
                wrapped, policy, runner = load_keyboard_policy(env, args.checkpoint, device=args.device, seed=args.seed)
                result.update(_time_env(env, wrapped, policy, runner, args))
                from isaaclab_newton.physics.newton_manager import NewtonManager

                result["cuda_graph_active"] = NewtonManager._graph is not None
            finally:
                env.close()
    except torch.OutOfMemoryError as error:
        result.update(status="oom", error=str(error)[:300])
    return result


def _time_env(env: Any, wrapped: Any, policy: Any, runner: Any, args: argparse.Namespace) -> dict[str, Any]:
    from isaaclab_neural.eval.eval import reset_policy

    with torch.inference_mode():
        env.reset()
        obs = wrapped.get_observations()

        def step(obs):
            obs, _, done, _ = wrapped.step(policy(obs))
            reset_policy(policy, runner, done)
            return obs, int(done.sum())

        for _ in range(args.warmup):
            obs, _ = step(obs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        policy_s = step_s = 0.0
        resets = 0
        for _ in range(args.steps):
            t0 = time.perf_counter()
            actions = policy(obs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            obs, _, done, _ = wrapped.step(actions)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            reset_policy(policy, runner, done)
            policy_s, step_s, resets = policy_s + t1 - t0, step_s + t2 - t1, resets + int(done.sum())
        # Phase 2: synchronize around each physics frame to split physics from the rest of the environment step.
        original_step, physics_s, total_s = env.sim.step, [0.0], 0.0

        def timed_sim_step(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = original_step(*a, **k)
            torch.cuda.synchronize()
            physics_s[0] += time.perf_counter() - t
            return out

        env.sim.step = timed_sim_step
        try:
            for _ in range(args.breakdown_steps):
                torch.cuda.synchronize()
                t = time.perf_counter()
                obs, _ = step(obs)
                torch.cuda.synchronize()
                total_s += time.perf_counter() - t
        finally:
            env.sim.step = original_step
        profile = None
        if args.nerd_checkpoint is not None and args.profile_steps > 0:
            from isaaclab_neural.utils import step_profile

            step_profile.enable(cuda_synchronize=True)
            step_profile.reset()
            try:
                for _ in range(args.profile_steps):
                    obs, _ = step(obs)
                profile = {k: 1000 * v / args.profile_steps for k, v in step_profile.summary_dict().items() if v > 0}
            finally:
                step_profile.disable()
    control_steps = args.num_envs * args.steps
    return {
        "status": "ok",
        "timed_control_steps_per_env": args.steps,
        "env_step_ms": 1000 * step_s / args.steps,
        "policy_ms": 1000 * policy_s / args.steps,
        "env_control_steps_per_s": control_steps / step_s,
        "env_control_steps_per_s_with_policy": control_steps / (step_s + policy_s),
        "physics_frames_per_s": control_steps * PHYSICS_FRAMES_PER_CONTROL_STEP / step_s,
        "realtime_factor": control_steps * PHYSICS_FRAMES_PER_CONTROL_STEP * env.physics_dt / step_s,
        "resets_during_timing": resets,
        "breakdown_env_step_ms": 1000 * total_s / args.breakdown_steps,
        "breakdown_physics_ms_per_control_step": 1000 * physics_s[0] / args.breakdown_steps,
        "breakdown_physics_fraction": physics_s[0] / total_s,
        "nerd_profile_ms_per_control_step": profile,
        "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main() -> None:
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    model = sub.add_parser("model", help="Model-only NeRD inference over a batch sweep.")
    model.add_argument("--nerd-checkpoint", type=Path, required=True)
    model.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 256, 1024, 2048, 4096, 8192])
    model.add_argument("--valid-tokens", type=int, default=24, help="Valid contact rows per world and frame.")
    model.add_argument("--autocast", choices=("fp32", "fp16", "bf16"), default="fp32")
    model.add_argument("--warmup", type=int, default=10)
    model.add_argument("--iterations", type=int, default=50)
    model.add_argument("--device", default="cuda:0")
    env = sub.add_parser("env", help="Policy-driven end-to-end environment steps at one world count.")
    env.add_argument("--nerd-checkpoint", type=Path, help="NeRD dynamics; omit for native MJWarp.")
    env.add_argument("--checkpoint", type=Path, default=Path("pretrained/control_policy/so101_keyboard/model_14999.pt"))
    env.add_argument("--robot-usd", type=Path)
    env.add_argument(
        "--frame-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint's NeRD use_frame_cache.",
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
        result = benchmark_model(args)
    else:
        args.device = args.device or "cuda:0"
        result = benchmark_env(args, args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print("BENCH", json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
