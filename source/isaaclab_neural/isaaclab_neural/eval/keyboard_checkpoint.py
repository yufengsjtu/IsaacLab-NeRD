# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a Keyboard NeRD checkpoint with cold-start, world-preserving action rollouts.

Only the initial physical/MDP state is restored. Prediction errors are scored up
through the first teacher or student terminal boundary, never after an auto-reset.
This diagnostic synchronizes and records tensors; its throughput is not an FPS benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, cast

import h5py
import torch
from isaaclab_neural.data.keyboard import (
    KEYBOARD_REPLAY_LAYOUT,
    keyboard_replay_batches,
    read_keyboard_metadata,
    validate_keyboard_runtime,
    validate_keyboard_training_contracts,
)
from isaaclab_neural.generate.keyboard_dataset import restore_keyboard_initial_state
from isaaclab_neural.utils.checkpoint import load_checkpoint


def rollout_batch(
    env: Any,
    expected: dict[str, torch.Tensor],
    context: dict[str, torch.Tensor],
    active_worlds: list[int],
    metadata: dict[str, Any],
    horizon: int,
) -> dict[str, Any]:
    """Measure state errors [m, rad], normal-force errors [N], and task event disagreements."""
    restore_keyboard_initial_state(env, context)
    adapter = env.neural_adapter
    adapter.sync(update_history=False)
    adapter.reset_history()
    validate_keyboard_runtime(metadata, env.scene, adapter.model)
    if adapter.solver.contact_filter == "solver_active_movable":
        fixed_slots = getattr(adapter.solver.contact_adapter, "fixed_body_slots", None)
        if (
            metadata.get("contact_filter") != "solver_active_movable"
            or fixed_slots is None
            or list(fixed_slots) != metadata.get("contact_fixed_body_slots")
        ):
            raise ValueError("Checkpoint runtime contact filtering differs from the recording contract.")
    records = []
    original_update = env.scene.update
    had_override = "update" in vars(env.scene)
    counter = env._sim_step_counter
    cache_error = 0.0
    original_solver_step = adapter.solver.step
    had_solver_override = "step" in vars(adapter.solver)
    neural_steps = 0

    def solver_step(*args, **kwargs):
        nonlocal neural_steps
        neural_steps += 1
        return original_solver_step(*args, **kwargs)

    def update(*args, **kwargs):
        nonlocal counter, cache_error
        original_update(*args, **kwargs)
        if env._sim_step_counter == counter:
            return
        if env._sim_step_counter - counter != 1:
            raise RuntimeError("Skipped a student physics frame.")
        counter = env._sim_step_counter
        forces = env.scene.sensors["robot_contact"].data.net_normal_forces_w_history.torch[:, 0].clone()
        cached = adapter.solver.predicted_robot_net_normal_forces_w
        cache_error = max(cache_error, float((forces - cached).abs().max()))
        keyboard = env.scene["keyboard"]
        command = env.command_manager.get_term("typing")
        q = keyboard.data.joint_pos.torch[command._inst_idx, command._col_idx]
        limits = keyboard.data.joint_pos_limits.torch[command._inst_idx, command._col_idx]
        pressed = q < limits[..., 1] - command.cfg.actuation_fraction * (limits[..., 1] - limits[..., 0])
        adapter.sync(update_history=False)
        records.append(
            {
                "next_states": adapter.states_torch.detach().cpu().clone(),
                "forces": forces.detach().cpu(),
                "pressed": pressed.detach().cpu(),
            }
        )

    env.scene.update = update
    adapter.solver.step = solver_step
    masks = []
    alive = torch.zeros(env.num_envs, dtype=torch.bool)
    alive[active_worlds] = True
    termination_mismatches = 0
    typed_mismatches = 0
    try:
        for start in range(0, horizon, env.cfg.decimation):
            actions = expected["actions"][:, start : start + env.cfg.decimation]
            if not torch.equal(actions, actions[:, :1].expand_as(actions)):
                raise ValueError("Recorded actions change inside a control interval.")
            before = len(records)
            result = env.step(actions[:, 0].to(env.device))
            if len(records) - before != env.cfg.decimation:
                raise RuntimeError("Student physics frame count differs from the action contract.")
            masks.extend([alive.clone()] * env.cfg.decimation)
            last = start + env.cfg.decimation - 1
            actual_done = (result[2] | result[3]).cpu()
            teacher_done = expected["terminated"][:, last] | expected["truncated"][:, last]
            termination_mismatches += int(((actual_done != teacher_done) & alive).sum())
            valid_command = alive & ~actual_done & ~teacher_done
            typed = env.command_manager.get_term("typing").typed.cpu()
            typed_mismatches += int(((typed != expected["typed_after"][:, last]) & valid_command[:, None]).sum())
            alive &= ~actual_done & ~teacher_done
    finally:
        if had_override:
            env.scene.update = original_update
        else:
            del env.scene.update
        if had_solver_override:
            adapter.solver.step = original_solver_step
        else:
            del adapter.solver.step
    if neural_steps != horizon:
        raise RuntimeError(f"Expected {horizon} learned physics steps, observed {neural_steps}.")
    actual = {key: torch.stack([record[key] for record in records], dim=1) for key in records[0]}
    mask = torch.stack(masks, dim=1)
    error_by_frame = (actual["next_states"] - expected["next_states"][:, :horizon]).abs()
    error = error_by_frame[mask]
    force_error = (actual["forces"] - expected["robot_net_normal_forces_w"][:, :horizon])[mask]
    nq = metadata["num_q"]
    keys = metadata["key_q_indices"]
    robot = sorted(set(range(nq)) - set(keys))
    pressed = actual["pressed"][mask]
    target_pressed = expected["key_pressed"][:, :horizon][mask]
    return {
        "neural_steps_per_world": neural_steps,
        "state_error_curves": {
            name: [
                {
                    "max": float(error_by_frame[mask[:, step], step][:, indices].max()),
                    "rmse": float(error_by_frame[mask[:, step], step][:, indices].square().mean().sqrt()),
                }
                if bool(mask[:, step].any())
                else None
                for step in range(horizon)
            ]
            for name, indices in {
                "robot_q_rad": robot,
                "key_q_m": keys,
                "robot_qd_rad_s": [nq + i for i in robot],
                "key_qd_m_s": [nq + i for i in keys],
            }.items()
        },
        "requested_physics_frames": len(active_worlds) * horizon,
        "scored_physics_frames": int(mask.sum()),
        "key_requested_positive_samples": int(expected["key_pressed"][active_worlds, :horizon].sum()),
        "finite": bool(torch.isfinite(error).all() and torch.isfinite(force_error).all()),
        "robot_q_max_rad": float(error[:, robot].max()),
        "key_q_max_m": float(error[:, keys].max()),
        "robot_qd_max_rad_s": float(error[:, [nq + i for i in robot]].max()),
        "key_qd_max_m_s": float(error[:, [nq + i for i in keys]].max()),
        "force_max_component_error_n": float(force_error.abs().max()),
        "force_vector_rmse_n": float(force_error.square().sum(-1).mean().sqrt()),
        "key_pressed_mismatches": int((pressed != target_pressed).sum()),
        "key_true_positives": int((pressed & target_pressed).sum()),
        "key_false_positives": int((pressed & ~target_pressed).sum()),
        "key_false_negatives": int((~pressed & target_pressed).sum()),
        "key_target_positive_samples": int(target_pressed.sum()),
        "termination_mismatches": termination_mismatches,
        "typed_mismatches": typed_mismatches,
        "sensor_cache_max_error_n": cache_error,
    }


def main() -> None:
    import gymnasium as gym
    from isaaclab_neural.envs.keyboard_nerd_env import NerdSO101KeyboardEnv, NerdSO101KeyboardEnvCfg
    from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    metadata = read_keyboard_metadata(args.dataset)
    if metadata.get("training_parts"):
        raise ValueError("A training view has no synchronous timeline; evaluate its original recorded parts.")
    checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    contracts = checkpoint["cfg"]["keyboard_dataset_contracts"]
    train_contracts = contracts.get("train_sources", {"train": contracts["train"]})
    validate_keyboard_training_contracts(train_contracts, {"evaluation": metadata})
    synchronous = metadata.get("replay_layout") == KEYBOARD_REPLAY_LAYOUT
    with h5py.File(args.dataset, "r") as file:
        data_group = file["replay/frames"] if synchronous else file["data"]
        context_group = file["replay/contexts"] if synchronous else file["context/trajectories"]
        assert isinstance(data_group, h5py.Group) and isinstance(context_group, h5py.Group)
        data = {
            name: torch.from_numpy(value[()]) for name, value in data_group.items() if isinstance(value, h5py.Dataset)
        }
        context = {
            name: torch.from_numpy(value[()])
            for name, value in context_group.items()
            if isinstance(value, h5py.Dataset)
        }
    if args.horizon <= 0 or args.horizon > data["states"].shape[1] or args.horizon % metadata["decimation"]:
        parser.error("horizon must fit the dataset and be a positive multiple of decimation")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("max-batches must be positive")
    cfg = NerdSO101KeyboardEnvCfg()
    cfg.scene.num_envs = metadata["source_num_envs"]
    cfg.seed = metadata["source_seed"]
    cfg.sim.device = args.device or "cuda:0"
    physics = cfg.sim.physics
    assert isinstance(physics, NerdNewtonCfg)
    physics.solver_cfg = NerdSolverCfg(**checkpoint["cfg"]["env"]["neural_solver_cfg"])
    physics.solver_cfg.neural_model_path = str(args.checkpoint.resolve())
    launch_cfg = copy.deepcopy(cfg)
    launch_cfg.sim.physics = NewtonCfg(num_substeps=1, use_cuda_graph=False)
    if synchronous:
        final_start = data["states"].shape[1] - args.horizon
        starts = sorted(set([*range(0, final_start + 1, args.horizon), final_start]))
        worlds = list(range(cfg.scene.num_envs))
        batches = [(worlds, worlds, start) for start in starts]
    else:
        batches = [
            (indices, worlds, 0)
            for indices, worlds in keyboard_replay_batches(
                context["source_env_id"], list(range(len(data["states"]))), cfg.scene.num_envs
            )
        ]
    reports = []
    with launch_simulation(launch_cfg, args):
        env = cast(NerdSO101KeyboardEnv, gym.make("IsaacContrib-Keyboard-SO101-NeRD", cfg=cfg).unwrapped)
        try:
            with torch.inference_mode():
                if env.neural_adapter.solver.dof_q_per_env != metadata["num_q"]:
                    raise ValueError("Checkpoint solver state layout differs from the dataset.")
                for indices, worlds, start in batches[: args.max_batches]:
                    report = rollout_batch(
                        env,
                        {key: value[indices, start : start + args.horizon] for key, value in data.items()},
                        {
                            key: value[indices, start // cfg.decimation] if synchronous else value[indices]
                            for key, value in context.items()
                        },
                        worlds,
                        metadata,
                        args.horizon,
                    )
                    report["source_worlds"] = worlds
                    if synchronous:
                        report["source_start_frame"] = start
                    else:
                        report["trajectory_indices"] = [indices[world] for world in worlds]
                    reports.append(report)
        finally:
            env.close()
    result = {
        "mode": "cold_start_nerd_action_rollout",
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "horizon": args.horizon,
        "source_layout": "synchronous_windows" if synchronous else "legacy_clips",
        "windows_may_overlap": synchronous and data["states"].shape[1] % args.horizon != 0,
        "batches": reports,
        "execution_passed": all(row["finite"] and row["sensor_cache_max_error_n"] <= 1e-6 for row in reports),
        "task_fidelity_accepted": False,
        "fps_benchmark": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["execution_passed"]:
        raise RuntimeError("Checkpoint execution or sensor publication failed; inspect the report.")


if __name__ == "__main__":
    main()
