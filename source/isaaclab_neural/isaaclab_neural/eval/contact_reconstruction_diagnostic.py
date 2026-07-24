# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare dataset contacts with contacts reconstructed after an explicit reset."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import h5py
import isaaclab_neural.envs  # noqa: F401 - registers built-in NeRD tasks
import numpy as np
import torch
import warp as wp
from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.solvers.neural_solver import NeuralSolver
from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint

from isaaclab_tasks.utils.hydra import hydra_task_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse diagnostic and IsaacLab launcher arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Registered NeRD task id.")
    parser.add_argument("--checkpoint", required=True, help="NeRD checkpoint used for one-step predictions.")
    parser.add_argument("--dataset", required=True, help="HDF5 trajectory dataset.")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of trajectories and runtime envs.")
    parser.add_argument("--trajectory-start", type=int, default=0, help="First trajectory index to compare.")
    parser.add_argument("--step", type=int, default=0, help="Trajectory step to compare.")

    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    return parser.parse_known_args()


def build_solver_cfg(checkpoint_path: str) -> NerdSolverCfg:
    """Reconstruct the solver configuration embedded in a checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    cfg = get_cfg_from_checkpoint(checkpoint, checkpoint_path)
    solver_cfg_dict = dict(cfg["env"]["neural_solver_cfg"])
    solver_cfg_dict.pop("use_cuda_graph", None)
    solver_cfg_dict["neural_model_path"] = checkpoint_path
    solver_cfg = NerdSolverCfg(**solver_cfg_dict)
    return solver_cfg


def configure_env(env_cfg, args: argparse.Namespace) -> None:
    """Apply diagnostic launch overrides."""
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = 0
    if args.device is not None:
        env_cfg.sim.device = args.device


def build_launch_cfg(env_cfg):
    """Return an upstream Newton config for launcher backend detection."""
    physics_cfg = env_cfg.sim.physics
    if not isinstance(physics_cfg, NerdNewtonCfg):
        return env_cfg

    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    launch_cfg = copy.deepcopy(env_cfg)
    launch_cfg.sim.physics = NewtonCfg(
        num_substeps=physics_cfg.num_substeps,
        debug_mode=physics_cfg.debug_mode,
        use_cuda_graph=False,
    )
    return launch_cfg


def load_dataset_batch(
    dataset_path: str,
    *,
    trajectory_start: int,
    num_envs: int,
    step: int,
    device: str,
    history_length: int = 1,
) -> dict[str, torch.Tensor]:
    """Load a history window ending at one timestep from consecutive trajectories."""
    if trajectory_start < 0:
        raise ValueError("trajectory_start must be non-negative.")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")

    path = Path(dataset_path).expanduser()
    with h5py.File(path, "r", swmr=True, libver="latest") as dataset_file:
        data_group = dataset_file["data"]
        base_required = {"states", "next_states", "joint_f", "root_body_q", "gravity_dir"}
        if "contact_tokens" in data_group:
            required = base_required | {"contact_tokens"}
        else:
            required = base_required | {
                "contact_masks",
                "contact_normals",
                "contact_depths",
                "contact_points_0",
                "contact_points_1",
                "contact_thicknesses_0",
                "contact_thicknesses_1",
            }
        missing = sorted(required - set(data_group.keys()))
        if missing:
            raise ValueError(f"Dataset is missing required diagnostic fields: {missing}.")
        trajectory_end = trajectory_start + num_envs
        states = data_group["states"]
        if trajectory_end > states.shape[0]:
            raise ValueError(f"Requested trajectory end {trajectory_end}, but dataset contains {states.shape[0]}.")
        if step < 0 or step >= states.shape[1]:
            raise ValueError(f"Step {step} is outside dataset trajectory length {states.shape[1]}.")
        if "traj_lengths" in data_group:
            traj_lengths = np.asarray(data_group["traj_lengths"][trajectory_start:trajectory_end])
            if np.any(step >= traj_lengths):
                raise ValueError(
                    f"Step {step} exceeds one or more selected trajectory lengths: {traj_lengths.tolist()}."
                )

        history_start = max(0, step - history_length + 1)
        batch = {}
        for key, dataset in data_group.items():
            if key == "traj_lengths":
                continue
            array = np.asarray(dataset[trajectory_start:trajectory_end, history_start : step + 1])
            if array.dtype == np.bool_:
                batch[key] = torch.as_tensor(array, dtype=torch.bool, device=device)
            else:
                batch[key] = torch.as_tensor(array, dtype=torch.float32, device=device)
    return batch


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Clone a flat tensor dictionary."""
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in batch.items()
    }


def preprocess_inputs(solver: NeuralSolver, raw_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply the same coordinate conversion and contact masking used during training."""
    inputs = clone_batch(raw_inputs)
    solver.process_neural_model_inputs(inputs)
    return inputs


@torch.no_grad()
def predict_next_states(
    solver: NeuralSolver,
    processed_inputs: dict[str, torch.Tensor],
    dt: float,
) -> torch.Tensor:
    """Run one model step and convert the prediction back to world coordinates."""
    model_inputs = clone_batch(processed_inputs)
    prediction = solver.neural_model(model_inputs, single_step=True).squeeze(1)
    next_states_model = solver.convert_prediction_to_next_states(
        states=processed_inputs["states"][:, -1],
        prediction=prediction,
        dt=dt,
    )
    next_states_world = solver.convert_states_back_to_world(processed_inputs["root_body_q"], next_states_model)
    solver.wrap2PI(next_states_world)
    return next_states_world


def state_error(solver: NeuralSolver, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return state, position, and velocity mean-squared errors."""
    difference = target - prediction
    solver.wrap2PI(difference)
    q_end = solver.dof_q_per_env
    return {
        "state_mse": float(difference.square().mean()),
        "q_mse": float(difference[:, :q_end].square().mean()),
        "qd_mse": float(difference[:, q_end:].square().mean()),
    }


def contact_metrics(
    dataset_inputs: dict[str, torch.Tensor],
    runtime_inputs: dict[str, torch.Tensor],
    num_contacts_per_env: int,
) -> dict[str, float]:
    """Compare body-frame contacts slot-wise and as unordered point sets."""
    dataset_mask = dataset_inputs["contact_masks"][:, -1].bool()
    runtime_mask = runtime_inputs["contact_masks"][:, -1].bool()
    dataset_points = dataset_inputs["contact_points_1"][:, -1].reshape(-1, num_contacts_per_env, 3)
    runtime_points = runtime_inputs["contact_points_1"][:, -1].reshape(-1, num_contacts_per_env, 3)

    same_active = dataset_mask & runtime_mask
    if same_active.any():
        slot_distance = torch.linalg.vector_norm(dataset_points - runtime_points, dim=-1)[same_active]
        slot_mean = float(slot_distance.mean())
        slot_max = float(slot_distance.max())
    else:
        slot_mean = float("nan")
        slot_max = float("nan")

    chamfer_values = []
    count_differences = []
    one_sided_empty_count = 0
    for env_index in range(dataset_mask.shape[0]):
        dataset_active = dataset_points[env_index, dataset_mask[env_index]]
        runtime_active = runtime_points[env_index, runtime_mask[env_index]]
        count_differences.append(abs(dataset_active.shape[0] - runtime_active.shape[0]))
        if dataset_active.shape[0] == 0 and runtime_active.shape[0] == 0:
            chamfer_values.append(torch.tensor(0.0, device=dataset_points.device))
        elif dataset_active.shape[0] == 0 or runtime_active.shape[0] == 0:
            chamfer_values.append(torch.tensor(float("inf"), device=dataset_points.device))
            one_sided_empty_count += 1
        else:
            distances = torch.cdist(dataset_active, runtime_active)
            chamfer_values.append(0.5 * (distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()))

    chamfer_tensor = torch.stack(chamfer_values)
    return {
        "mask_agreement": float((dataset_mask == runtime_mask).float().mean()),
        "dataset_active_mean": float(dataset_mask.sum(dim=-1).float().mean()),
        "runtime_active_mean": float(runtime_mask.sum(dim=-1).float().mean()),
        "active_count_abs_diff_mean": float(torch.tensor(count_differences, dtype=torch.float32).mean()),
        "slot_point_distance_mean": slot_mean,
        "slot_point_distance_max": slot_max,
        "set_chamfer_distance_mean": float(chamfer_tensor.mean()),
        "one_sided_empty_count": float(one_sided_empty_count),
        "one_sided_empty_fraction": one_sided_empty_count / max(dataset_mask.shape[0], 1),
    }


def contact_token_metrics(
    dataset_inputs: dict[str, torch.Tensor],
    runtime_inputs: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compare valid contact token counts and geometry for the latest frame."""
    dataset_tokens = dataset_inputs["contact_tokens"][:, -1]
    runtime_tokens = runtime_inputs["contact_tokens"][:, -1]
    dataset_valid = dataset_tokens[..., 0] > 0.5
    runtime_valid = runtime_tokens[..., 0] > 0.5
    geometry_slice = slice(4, 17)
    if (dataset_valid & runtime_valid).any():
        geometry_diff = dataset_tokens[..., geometry_slice] - runtime_tokens[..., geometry_slice]
        geometry_rmse = float(geometry_diff[dataset_valid & runtime_valid].square().mean().sqrt())
    else:
        geometry_rmse = float("nan")
    metrics = {
        "valid_count_abs_diff_mean": float((dataset_valid.sum(dim=-1) - runtime_valid.sum(dim=-1)).abs().float().mean()),
        "geometry_rmse": geometry_rmse,
    }
    if "contact_token_overflow" in runtime_inputs:
        metrics["runtime_overflow_max"] = float(runtime_inputs["contact_token_overflow"][:, -1].max())
    return metrics


def input_difference_metrics(
    dataset_inputs: dict[str, torch.Tensor],
    runtime_inputs: dict[str, torch.Tensor],
    num_contacts_per_env: int,
) -> dict[str, float]:
    """Report per-field differences after model-input preprocessing."""
    metrics = {}
    dataset_mask = dataset_inputs["contact_masks"][:, -1].bool()
    runtime_mask = runtime_inputs["contact_masks"][:, -1].bool()
    active_mask = dataset_mask & runtime_mask

    for key in ("states", "root_body_q", "joint_f", "gravity_dir"):
        difference = (dataset_inputs[key] - runtime_inputs[key]).abs()
        metrics[f"{key}_abs_mean"] = float(difference.mean())
        metrics[f"{key}_abs_max"] = float(difference.max())

    for key in ("contact_points_0", "contact_points_1", "contact_normals"):
        dataset_value = dataset_inputs[key][:, -1].reshape(-1, num_contacts_per_env, 3)
        runtime_value = runtime_inputs[key][:, -1].reshape(-1, num_contacts_per_env, 3)
        difference = torch.linalg.vector_norm(dataset_value - runtime_value, dim=-1)
        active_difference = difference[active_mask]
        metrics[f"{key}_l2_mean_active"] = (
            float(active_difference.mean()) if active_difference.numel() else float("nan")
        )
        metrics[f"{key}_l2_max_active"] = float(active_difference.max()) if active_difference.numel() else float("nan")

    for key in ("contact_depths", "contact_thicknesses_0", "contact_thicknesses_1"):
        difference = (dataset_inputs[key][:, -1] - runtime_inputs[key][:, -1]).abs()
        active_difference = difference[active_mask]
        metrics[f"{key}_abs_mean_active"] = (
            float(active_difference.mean()) if active_difference.numel() else float("nan")
        )
        metrics[f"{key}_abs_max_active"] = float(active_difference.max()) if active_difference.numel() else float("nan")
    return metrics


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    """Print a compact metric group."""
    print(f"\n[{title}]")
    for key, value in metrics.items():
        print(f"{key}={value:.8g}")


def run_diagnostic(env, args: argparse.Namespace) -> None:
    """Run dataset-contact and runtime-contact one-step A/B predictions."""
    adapter = env.neural_adapter
    solver = adapter.solver
    solver.eval()

    dataset_raw = load_dataset_batch(
        args.dataset,
        trajectory_start=args.trajectory_start,
        num_envs=args.num_envs,
        step=args.step,
        device=str(solver.torch_device),
        history_length=int(getattr(solver, "num_states_history", 1)),
    )
    if "contact_tokens" in dataset_raw:
        max_tokens = dataset_raw["contact_tokens"].shape[-2]
        runtime_capacity = int(getattr(solver, "max_contact_tokens", 0) or solver.num_contacts_per_env)
        if max_tokens != runtime_capacity:
            raise ValueError(
                f"Dataset contact tokens ({max_tokens}) do not match runtime capacity ({runtime_capacity})."
            )
    elif dataset_raw["contact_masks"].shape[-1] != solver.num_contacts_per_env:
        raise ValueError(
            f"Dataset contact slots ({dataset_raw['contact_masks'].shape[-1]}) do not match "
            f"runtime slots ({solver.num_contacts_per_env})."
        )
    target_next_states = dataset_raw["next_states"][:, -1].clone()
    dt = float(getattr(env, "step_dt", getattr(env, "physics_dt")))

    dataset_processed = preprocess_inputs(solver, dataset_raw)
    dataset_prediction = predict_next_states(solver, dataset_processed, dt)

    adapter.reset(initial_states=dataset_raw["states"][:, -1])
    manager = adapter.manager
    manager._control.joint_f.assign(wp.from_torch(dataset_raw["joint_f"][:, -1].reshape(-1)))
    adapter.sync(update_history=False)
    runtime_current = NeuralSolver.get_raw_neural_model_inputs(solver)
    runtime_raw = clone_batch(dataset_raw)
    for key, value in runtime_current.items():
        if key in runtime_raw:
            runtime_raw[key][:, -1:] = value
    runtime_processed = preprocess_inputs(solver, runtime_raw)
    runtime_prediction = predict_next_states(solver, runtime_processed, dt)

    print(
        "[setup] "
        f"dataset={args.dataset}, checkpoint={args.checkpoint}, num_envs={args.num_envs}, "
        f"trajectory_start={args.trajectory_start}, step={args.step}, "
        f"history_length={dataset_raw['states'].shape[1]}"
    )
    print_metrics("dataset_contact_prediction", state_error(solver, dataset_prediction, target_next_states))
    print_metrics("runtime_reconstructed_prediction", state_error(solver, runtime_prediction, target_next_states))
    print_metrics(
        "contact_comparison",
        contact_token_metrics(dataset_processed, runtime_processed)
        if "contact_tokens" in dataset_processed
        else contact_metrics(dataset_processed, runtime_processed, solver.num_contacts_per_env),
    )
    if "contact_tokens" not in dataset_processed:
        print_metrics(
            "input_differences",
            input_difference_metrics(dataset_processed, runtime_processed, solver.num_contacts_per_env),
        )
    print_metrics(
        "prediction_difference",
        state_error(solver, runtime_prediction, dataset_prediction),
    )


def main() -> None:
    """Parse arguments, launch the NeRD task, and run the A/B diagnostic."""
    args_cli, hydra_args = parse_args()
    sys.argv = [sys.argv[0]] + hydra_args

    @hydra_task_config(args_cli.task, "")
    def hydra_main(env_cfg, _agent_cfg=None) -> None:
        configure_env(env_cfg, args_cli)
        solver_cfg = build_solver_cfg(args_cli.checkpoint)

        from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

        from isaaclab_tasks.utils import launch_simulation

        with launch_simulation(build_launch_cfg(env_cfg), args_cli):
            import gymnasium as gym

            with newton_material_binding_api_autofix():
                env = gym.make(args_cli.task, cfg=env_cfg, device=args_cli.device, solver_cfg=solver_cfg).unwrapped
            try:
                run_diagnostic(env, args_cli)
            finally:
                env.close()

    hydra_main()  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
