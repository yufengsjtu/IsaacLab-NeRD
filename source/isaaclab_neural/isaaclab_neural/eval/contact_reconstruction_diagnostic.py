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
from isaaclab_neural.data import (
    build_terrain_context,
    read_terrain_context,
    set_terrain_seed,
    validate_terrain_context,
)
from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.solvers.neural_solver import NeuralSolver
from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint

from isaaclab_tasks.utils.hydra import hydra_task_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse diagnostic and IsaacLab launcher arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Registered NeRD task id.")
    parser.add_argument("--checkpoint", help="Optional NeRD checkpoint used for one-step predictions.")
    parser.add_argument("--dataset", required=True, help="HDF5 trajectory dataset.")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of trajectories and runtime envs.")
    parser.add_argument("--trajectory-start", type=int, default=0, help="First trajectory index to compare.")
    parser.add_argument("--step", type=int, default=0, help="Trajectory step to compare.")
    parser.add_argument(
        "--require-terrain-context",
        action="store_true",
        help="Require terrain provenance, restore trajectory patches, and fail on contact mismatch.",
    )
    parser.add_argument(
        "--terrain-seed-override",
        type=int,
        default=None,
        help="Override the recorded terrain seed for a negative consistency test.",
    )
    parser.add_argument(
        "--contact-tolerance",
        type=float,
        default=1.0e-4,
        help="Maximum accepted reconstructed contact-point Chamfer distance.",
    )
    parser.add_argument("--num-contacts-per-env", type=int, default=64)
    parser.add_argument("--contact-mode", choices=("fixed_ground", "newton_native"), default="newton_native")
    parser.add_argument(
        "--contact-packing-policy",
        choices=("stable_index", "penetration_priority", "random", "force_priority"),
        default="penetration_priority",
    )
    parser.add_argument("--states-frame", choices=("world", "body", "body_translation_only"), default="body")

    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    return parser.parse_known_args()


def build_solver_cfg(checkpoint_path: str | None, args: argparse.Namespace) -> NerdSolverCfg:
    """Build a diagnostic solver, optionally loading model settings from a checkpoint."""
    if checkpoint_path is None:
        return NerdSolverCfg(
            name="NeuralSolver",
            states_frame=args.states_frame,
            states_embedding_type="identical",
            prediction_type="relative",
            orientation_prediction_parameterization="quaternion",
            min_contact_event_threshold=0.12,
            num_contacts_per_env=args.num_contacts_per_env,
            contact_mode=args.contact_mode,
            contact_packing_policy=args.contact_packing_policy,
        )
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    cfg = get_cfg_from_checkpoint(checkpoint, checkpoint_path)
    solver_cfg_dict = dict(cfg["env"]["neural_solver_cfg"])
    solver_cfg_dict.pop("use_cuda_graph", None)
    solver_cfg_dict["neural_model_path"] = checkpoint_path
    solver_cfg = NerdSolverCfg(**solver_cfg_dict)
    return solver_cfg


def configure_env(env_cfg, args: argparse.Namespace) -> dict[str, Any] | None:
    """Apply diagnostic launch overrides."""
    env_cfg.scene.num_envs = args.num_envs
    terrain_context = read_terrain_context(args.dataset)
    if terrain_context is None:
        if args.require_terrain_context:
            raise ValueError(f"Dataset {args.dataset!r} has no terrain context.")
        env_cfg.seed = 0
    else:
        terrain_seed = (
            int(args.terrain_seed_override)
            if args.terrain_seed_override is not None
            else int(terrain_context["seed"])
        )
        set_terrain_seed(env_cfg, terrain_seed)
    if args.device is not None:
        env_cfg.sim.device = args.device
    return terrain_context


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
        required = {
            "states",
            "next_states",
            "joint_f",
            "root_body_q",
            "gravity_dir",
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
        if "context" in dataset_file and "trajectories" in dataset_file["context"]:
            trajectory_group = dataset_file["context"]["trajectories"]
            for key, dataset in trajectory_group.items():
                array = np.asarray(dataset[trajectory_start:trajectory_end])
                batch[key] = torch.as_tensor(array, device=device)
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


def restore_terrain_patch(env, batch: dict[str, torch.Tensor], *, required: bool) -> None:
    """Restore recorded terrain patch assignments for the selected trajectories."""
    keys = {"terrain_level", "terrain_type", "env_origin"}
    if not keys.issubset(batch):
        if required:
            raise ValueError(f"Dataset is missing trajectory terrain context: {sorted(keys - set(batch))}.")
        return

    terrain = getattr(getattr(env.neural_adapter.isaaclab_env, "scene", None), "terrain", None)
    if terrain is None:
        if required:
            raise ValueError("The diagnostic task has no terrain importer.")
        return
    num_envs = batch["states"].shape[0]
    terrain.terrain_levels[:num_envs].copy_(batch["terrain_level"].to(terrain.device, dtype=torch.long))
    terrain.terrain_types[:num_envs].copy_(batch["terrain_type"].to(terrain.device, dtype=torch.long))
    terrain.env_origins[:num_envs].copy_(batch["env_origin"].to(terrain.device, dtype=torch.float32))


def validate_contact_metrics(metrics: dict[str, float], tolerance: float) -> None:
    """Fail strict diagnostics when reconstructed contacts differ materially."""
    if metrics["mask_agreement"] != 1.0:
        raise ValueError(f"Contact-mask agreement is {metrics['mask_agreement']:.8g}, expected 1.0.")
    chamfer = metrics["set_chamfer_distance_mean"]
    if not np.isfinite(chamfer) or chamfer > tolerance:
        raise ValueError(f"Contact-point Chamfer distance {chamfer:.8g} exceeds tolerance {tolerance:.8g}.")


def run_diagnostic(env, args: argparse.Namespace) -> None:
    """Run dataset-contact and runtime-contact one-step A/B predictions."""
    adapter = env.neural_adapter
    solver = adapter.solver
    if solver.neural_model is not None:
        solver.eval()

    dataset_raw = load_dataset_batch(
        args.dataset,
        trajectory_start=args.trajectory_start,
        num_envs=args.num_envs,
        step=args.step,
        device=str(solver.torch_device),
        history_length=int(getattr(solver, "num_states_history", 1)),
    )
    if dataset_raw["contact_masks"].shape[-1] != solver.num_contacts_per_env:
        raise ValueError(
            f"Dataset contact slots ({dataset_raw['contact_masks'].shape[-1]}) do not match "
            f"runtime slots ({solver.num_contacts_per_env})."
        )
    target_next_states = dataset_raw["next_states"][:, -1].clone()
    dt = float(getattr(env, "step_dt", getattr(env, "physics_dt")))

    dataset_processed = preprocess_inputs(solver, dataset_raw)

    restore_terrain_patch(env, dataset_raw, required=args.require_terrain_context)
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

    print(
        "[setup] "
        f"dataset={args.dataset}, checkpoint={args.checkpoint or 'none'}, num_envs={args.num_envs}, "
        f"trajectory_start={args.trajectory_start}, step={args.step}, "
        f"history_length={dataset_raw['states'].shape[1]}"
    )
    comparison_metrics = contact_metrics(dataset_processed, runtime_processed, solver.num_contacts_per_env)
    print_metrics("contact_comparison", comparison_metrics)
    print_metrics(
        "input_differences",
        input_difference_metrics(dataset_processed, runtime_processed, solver.num_contacts_per_env),
    )
    if solver.neural_model is not None:
        dataset_prediction = predict_next_states(solver, dataset_processed, dt)
        runtime_prediction = predict_next_states(solver, runtime_processed, dt)
        print_metrics("dataset_contact_prediction", state_error(solver, dataset_prediction, target_next_states))
        print_metrics("runtime_reconstructed_prediction", state_error(solver, runtime_prediction, target_next_states))
        print_metrics(
            "prediction_difference",
            state_error(solver, runtime_prediction, dataset_prediction),
        )
    if args.require_terrain_context:
        validate_contact_metrics(comparison_metrics, args.contact_tolerance)
        print("[strict-context] PASS")


def main() -> None:
    """Parse arguments, launch the NeRD task, and run the A/B diagnostic."""
    args_cli, hydra_args = parse_args()
    sys.argv = [sys.argv[0]] + hydra_args

    @hydra_task_config(args_cli.task, "")
    def hydra_main(env_cfg, _agent_cfg=None) -> None:
        expected_terrain_context = configure_env(env_cfg, args_cli)
        solver_cfg = build_solver_cfg(args_cli.checkpoint, args_cli)

        from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

        from isaaclab_tasks.utils import launch_simulation

        with launch_simulation(build_launch_cfg(env_cfg), args_cli):
            import gymnasium as gym

            if expected_terrain_context is not None:
                actual_terrain_context = build_terrain_context(
                    env_cfg,
                    int(
                        args_cli.terrain_seed_override
                        if args_cli.terrain_seed_override is not None
                        else expected_terrain_context["seed"]
                    ),
                )
                if actual_terrain_context is None:
                    raise ValueError("Dataset has terrain context but the diagnostic task has no terrain generator.")
                validate_terrain_context(expected_terrain_context, actual_terrain_context)
                print(
                    "[terrain-context] "
                    f"seed={expected_terrain_context['seed']}, "
                    f"mesh_sha256={expected_terrain_context['mesh_sha256']}, PASS"
                )

            with newton_material_binding_api_autofix():
                env = gym.make(args_cli.task, cfg=env_cfg, device=args_cli.device, solver_cfg=solver_cfg).unwrapped
            try:
                run_diagnostic(env, args_cli)
            finally:
                env.close()

    hydra_main()  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
