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
from typing import Any, cast

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
from isaaclab_neural.eval.contact_set_matching import (
    contact_set_matching_metrics,
    contact_set_tolerance_failures,
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
        "--random-samples",
        type=int,
        default=0,
        help="Random trajectory windows to compare in batches of --num-envs. Disabled when 0.",
    )
    parser.add_argument("--random-seed", type=int, default=0, help="Seed for random trajectory-window sampling.")
    parser.add_argument(
        "--eval-horizon",
        type=int,
        default=10,
        help="Reserve this many post-reset steps when sampling random windows.",
    )
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
        help="Maximum accepted matched contact-point and depth error.",
    )
    parser.add_argument(
        "--contact-normal-tolerance",
        type=float,
        default=1.0e-3,
        help="Maximum accepted matched contact-normal L2 error.",
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
        data_group = cast(h5py.Group, dataset_file["data"])
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
        states = cast(h5py.Dataset, data_group["states"])
        if trajectory_end > states.shape[0]:
            raise ValueError(f"Requested trajectory end {trajectory_end}, but dataset contains {states.shape[0]}.")
        if step < 0 or step >= states.shape[1]:
            raise ValueError(f"Step {step} is outside dataset trajectory length {states.shape[1]}.")
        if "traj_lengths" in data_group:
            traj_lengths = np.asarray(
                cast(h5py.Dataset, data_group["traj_lengths"])[trajectory_start:trajectory_end]
            )
            if np.any(step >= traj_lengths):
                raise ValueError(
                    f"Step {step} exceeds one or more selected trajectory lengths: {traj_lengths.tolist()}."
                )

        history_start = max(0, step - history_length + 1)
        batch = {}
        for key, dataset in data_group.items():
            if key == "traj_lengths":
                continue
            dataset = cast(h5py.Dataset, dataset)
            array = np.asarray(dataset[trajectory_start:trajectory_end, history_start : step + 1])
            if array.dtype == np.bool_:
                batch[key] = torch.as_tensor(array, dtype=torch.bool, device=device)
            else:
                batch[key] = torch.as_tensor(array, dtype=torch.float32, device=device)
        if "context" in dataset_file:
            context_group = cast(h5py.Group, dataset_file["context"])
            if "trajectories" in context_group:
                trajectory_group = cast(h5py.Group, context_group["trajectories"])
                for key, dataset in trajectory_group.items():
                    dataset = cast(h5py.Dataset, dataset)
                    array = np.asarray(dataset[trajectory_start:trajectory_end])
                    batch[key] = torch.as_tensor(array, device=device)
    return batch


def load_random_dataset_batch(
    dataset_path: str,
    *,
    num_envs: int,
    history_length: int,
    eval_horizon: int,
    rng: np.random.Generator,
    device: str,
    excluded_trajectory_indices: set[int] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Load random evaluator-compatible history windows."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if history_length <= 0:
        raise ValueError("history_length must be positive.")
    if eval_horizon <= 0:
        raise ValueError("eval_horizon must be positive.")

    path = Path(dataset_path).expanduser()
    with h5py.File(path, "r", swmr=True, libver="latest") as dataset_file:
        data_group = cast(h5py.Group, dataset_file["data"])
        states = cast(h5py.Dataset, data_group["states"])
        if "traj_lengths" in data_group:
            traj_lengths = np.asarray(cast(h5py.Dataset, data_group["traj_lengths"])).astype(np.int64)
        else:
            traj_lengths = np.full(states.shape[0], states.shape[1], dtype=np.int64)
        eligible = np.flatnonzero(traj_lengths >= history_length + eval_horizon - 1)
        if excluded_trajectory_indices:
            excluded = np.fromiter(excluded_trajectory_indices, dtype=np.int64)
            eligible = eligible[~np.isin(eligible, excluded)]
        if eligible.size == 0:
            if excluded_trajectory_indices:
                raise ValueError("No unsampled evaluator-compatible trajectories remain.")
            raise ValueError(
                f"No trajectories are long enough for history_length={history_length} "
                f"and eval_horizon={eval_horizon}."
            )
        if excluded_trajectory_indices is not None and eligible.size < num_envs:
            raise ValueError(
                f"Requested {num_envs} additional unique trajectories, but only {eligible.size} remain eligible."
            )

        trajectory_indices = rng.choice(
            eligible,
            size=num_envs,
            replace=excluded_trajectory_indices is None,
        )
        steps = np.asarray(
            [
                rng.integers(history_length - 1, int(traj_lengths[trajectory_index]) - eval_horizon + 1)
                for trajectory_index in trajectory_indices
            ],
            dtype=np.int64,
        )
        batch = {}
        for key, dataset in data_group.items():
            if key == "traj_lengths":
                continue
            dataset = cast(h5py.Dataset, dataset)
            arrays = [
                np.asarray(dataset[trajectory_index, step - history_length + 1 : step + 1])
                for trajectory_index, step in zip(trajectory_indices, steps, strict=True)
            ]
            array = np.stack(arrays)
            if array.dtype == np.bool_:
                batch[key] = torch.as_tensor(array, dtype=torch.bool, device=device)
            else:
                batch[key] = torch.as_tensor(array, dtype=torch.float32, device=device)
        if "context" in dataset_file:
            context_group = cast(h5py.Group, dataset_file["context"])
            if "trajectories" in context_group:
                trajectory_group = cast(h5py.Group, context_group["trajectories"])
                for key, dataset in trajectory_group.items():
                    dataset = cast(h5py.Dataset, dataset)
                    array = np.stack([np.asarray(dataset[index]) for index in trajectory_indices])
                    batch[key] = torch.as_tensor(array, device=device)

    metadata = {
        "trajectory_index": trajectory_indices,
        "step": steps,
    }
    return batch, metadata


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
    if solver.neural_model is None:
        raise ValueError("A neural model is required for prediction diagnostics.")
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
    deduplication_tolerance: float = 1.0e-4,
    normal_tolerance: float = 1.0e-3,
) -> dict[str, float]:
    """Compare body-frame contacts slot-wise and as unordered point sets."""
    dataset_mask = dataset_inputs["contact_masks"][:, -1].bool()
    runtime_mask = runtime_inputs["contact_masks"][:, -1].bool()
    mask_mismatch = dataset_mask != runtime_mask
    dataset_points = dataset_inputs["contact_points_1"][:, -1].reshape(-1, num_contacts_per_env, 3)
    runtime_points = runtime_inputs["contact_points_1"][:, -1].reshape(-1, num_contacts_per_env, 3)

    same_active = dataset_mask & runtime_mask
    shared_active_slot_count = int(same_active.sum())
    if same_active.any():
        slot_distance = torch.linalg.vector_norm(dataset_points - runtime_points, dim=-1)[same_active]
        slot_mean = float(slot_distance.mean())
        slot_max = float(slot_distance.max())
        slot_sum = float(slot_distance.sum())
    else:
        slot_mean = float("nan")
        slot_max = float("nan")
        slot_sum = 0.0

    count_differences = []
    for env_index in range(dataset_mask.shape[0]):
        dataset_active = dataset_points[env_index, dataset_mask[env_index]]
        runtime_active = runtime_points[env_index, runtime_mask[env_index]]
        count_differences.append(abs(dataset_active.shape[0] - runtime_active.shape[0]))
    set_inputs_dataset = {
        key: dataset_inputs[key][:, -1]
        for key in (
            "contact_masks",
            "contact_points_0",
            "contact_points_1",
            "contact_normals",
            "contact_depths",
            "contact_thicknesses_0",
            "contact_thicknesses_1",
        )
    }
    set_inputs_runtime = {
        key: runtime_inputs[key][:, -1]
        for key in (
            "contact_masks",
            "contact_points_0",
            "contact_points_1",
            "contact_normals",
            "contact_depths",
            "contact_thicknesses_0",
            "contact_thicknesses_1",
        )
    }
    metrics = {
        "mask_agreement": float((dataset_mask == runtime_mask).float().mean()),
        "mask_mismatch_count": float(mask_mismatch.sum()),
        "mask_total_count": float(mask_mismatch.numel()),
        "mismatched_env_count": float(mask_mismatch.any(dim=-1).sum()),
        "dataset_active_mean": float(dataset_mask.sum(dim=-1).float().mean()),
        "runtime_active_mean": float(runtime_mask.sum(dim=-1).float().mean()),
        "active_count_abs_diff_mean": float(torch.tensor(count_differences, dtype=torch.float32).mean()),
        "slot_point_distance_mean": slot_mean,
        "slot_point_distance_max": slot_max,
        "slot_point_distance_sum": slot_sum,
        "shared_active_slot_count": float(shared_active_slot_count),
    }
    metrics.update(
        contact_set_matching_metrics(
            set_inputs_dataset,
            set_inputs_runtime,
            num_contacts_per_env,
            deduplication_tolerance=deduplication_tolerance,
            normal_tolerance=normal_tolerance,
        )
    )
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


def validate_contact_metrics(
    metrics: dict[str, float],
    tolerance: float,
    normal_tolerance: float = 1.0e-3,
) -> None:
    """Fail strict diagnostics only when unordered contact geometry differs."""
    failures = contact_set_tolerance_failures(
        metrics,
        point_tolerance=tolerance,
        normal_tolerance=normal_tolerance,
    )
    if failures:
        raise ValueError("Contact-set reconstruction mismatch: " + "; ".join(failures) + ".")


def print_contact_mismatch_examples(
    dataset_inputs: dict[str, torch.Tensor],
    runtime_inputs: dict[str, torch.Tensor],
    dataset_raw: dict[str, torch.Tensor],
    metadata: dict[str, np.ndarray],
) -> None:
    """Print trajectory and terrain coordinates for mismatched contact masks."""
    dataset_mask = dataset_inputs["contact_masks"][:, -1].bool()
    runtime_mask = runtime_inputs["contact_masks"][:, -1].bool()
    mismatch = dataset_mask != runtime_mask
    mismatched_envs = torch.nonzero(mismatch.any(dim=-1), as_tuple=False).flatten()
    for env_index in mismatched_envs[:8].tolist():
        slots = torch.nonzero(mismatch[env_index], as_tuple=False).flatten().tolist()
        terrain_level = int(dataset_raw.get("terrain_level", torch.full((dataset_mask.shape[0],), -1))[env_index])
        terrain_type = int(dataset_raw.get("terrain_type", torch.full((dataset_mask.shape[0],), -1))[env_index])
        print(
            "[contact-mismatch] "
            f"trajectory={int(metadata['trajectory_index'][env_index])}, "
            f"step={int(metadata['step'][env_index])}, terrain=({terrain_level}, {terrain_type}), "
            f"slots={slots[:8]}, dataset_active={int(dataset_mask[env_index].sum())}, "
            f"runtime_active={int(runtime_mask[env_index].sum())}"
        )


def compare_diagnostic_batch(
    env,
    args: argparse.Namespace,
    dataset_raw: dict[str, torch.Tensor],
    metadata: dict[str, np.ndarray],
    *,
    print_all_metrics: bool,
    run_predictions: bool,
) -> dict[str, float]:
    """Reset one runtime batch and compare it with recorded contacts."""
    adapter = env.neural_adapter
    solver = adapter.solver
    if solver.neural_model is not None:
        solver.eval()

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

    comparison_metrics = contact_metrics(
        dataset_processed,
        runtime_processed,
        solver.num_contacts_per_env,
        deduplication_tolerance=args.contact_tolerance,
        normal_tolerance=args.contact_normal_tolerance,
    )
    if print_all_metrics or comparison_metrics["mask_mismatch_count"] > 0:
        print_metrics("contact_comparison", comparison_metrics)
        print_metrics(
            "input_differences",
            input_difference_metrics(dataset_processed, runtime_processed, solver.num_contacts_per_env),
        )
    if comparison_metrics["mask_mismatch_count"] > 0:
        print_contact_mismatch_examples(dataset_processed, runtime_processed, dataset_raw, metadata)
    if run_predictions and solver.neural_model is not None:
        dataset_prediction = predict_next_states(solver, dataset_processed, dt)
        runtime_prediction = predict_next_states(solver, runtime_processed, dt)
        print_metrics("dataset_contact_prediction", state_error(solver, dataset_prediction, target_next_states))
        print_metrics("runtime_reconstructed_prediction", state_error(solver, runtime_prediction, target_next_states))
        print_metrics(
            "prediction_difference",
            state_error(solver, runtime_prediction, dataset_prediction),
        )
    return comparison_metrics


def aggregate_contact_metrics(metrics_by_batch: list[dict[str, float]], batch_size: int) -> dict[str, float]:
    """Aggregate equal-sized random diagnostic batches."""
    mismatch_count = sum(metrics["mask_mismatch_count"] for metrics in metrics_by_batch)
    mask_total = sum(metrics["mask_total_count"] for metrics in metrics_by_batch)
    mean_keys = (
        "dataset_active_mean",
        "runtime_active_mean",
        "active_count_abs_diff_mean",
        "set_chamfer_distance_mean",
        "set_hausdorff_distance_mean",
        "dataset_unique_active_mean",
        "runtime_unique_active_mean",
        "unique_count_abs_diff_mean",
        "one_sided_empty_fraction",
    )
    max_keys = (
        "set_hausdorff_distance_max",
        "matched_point0_distance_max",
        "matched_normal_l2_max",
        "matched_depth_abs_max",
        "matched_thickness0_abs_max",
        "matched_thickness1_abs_max",
        "unique_count_abs_diff_max",
    )
    slot_max_values = np.asarray(
        [
            metrics["slot_point_distance_max"]
            for metrics in metrics_by_batch
            if metrics["shared_active_slot_count"] > 0
        ],
        dtype=np.float64,
    )
    summary = {
        "mask_agreement": 1.0 - mismatch_count / max(mask_total, 1.0),
        "mask_mismatch_count": mismatch_count,
        "mask_total_count": mask_total,
        "mismatched_env_count": sum(metrics["mismatched_env_count"] for metrics in metrics_by_batch),
        "slot_point_distance_max": (
            float("nan")
            if slot_max_values.size == 0 or np.isnan(slot_max_values).any()
            else float(slot_max_values.max())
        ),
        "one_sided_empty_count": sum(metrics["one_sided_empty_count"] for metrics in metrics_by_batch),
        "slot_point_distance_sum": sum(metrics["slot_point_distance_sum"] for metrics in metrics_by_batch),
        "shared_active_slot_count": sum(metrics["shared_active_slot_count"] for metrics in metrics_by_batch),
        "num_samples": float(len(metrics_by_batch) * batch_size),
    }
    for key in mean_keys:
        values = np.asarray([metrics[key] for metrics in metrics_by_batch], dtype=np.float64)
        summary[key] = float(values.mean())
    summary["slot_point_distance_mean"] = (
        summary["slot_point_distance_sum"] / summary["shared_active_slot_count"]
        if summary["shared_active_slot_count"] > 0
        else float("nan")
    )
    for key in max_keys:
        values = np.asarray([metrics[key] for metrics in metrics_by_batch], dtype=np.float64)
        summary[key] = float("nan") if np.isnan(values).any() else float(values.max())
    return summary


def run_diagnostic(env, args: argparse.Namespace) -> None:
    """Run dataset-contact and runtime-contact one-step A/B predictions."""
    solver = env.neural_adapter.solver
    history_length = int(getattr(solver, "num_states_history", 1))
    print(
        "[setup] "
        f"dataset={args.dataset}, checkpoint={args.checkpoint or 'none'}, num_envs={args.num_envs}, "
        f"history_length={history_length}, random_samples={args.random_samples}"
    )
    if args.random_samples > 0:
        if args.random_samples % args.num_envs != 0:
            raise ValueError("--random-samples must be divisible by --num-envs.")
        rng = np.random.default_rng(args.random_seed)
        metrics_by_batch = []
        sampled_trajectory_indices: set[int] = set()
        for batch_index in range(args.random_samples // args.num_envs):
            dataset_raw, metadata = load_random_dataset_batch(
                args.dataset,
                num_envs=args.num_envs,
                history_length=history_length,
                eval_horizon=args.eval_horizon,
                rng=rng,
                device=str(solver.torch_device),
                excluded_trajectory_indices=sampled_trajectory_indices,
            )
            sampled_trajectory_indices.update(int(index) for index in metadata["trajectory_index"])
            metrics = compare_diagnostic_batch(
                env,
                args,
                dataset_raw,
                metadata,
                print_all_metrics=False,
                run_predictions=False,
            )
            metrics_by_batch.append(metrics)
            print(
                f"[random-progress] batch={batch_index + 1}/{args.random_samples // args.num_envs}, "
                f"mask_mismatches={int(metrics['mask_mismatch_count'])}"
            )
        summary = aggregate_contact_metrics(metrics_by_batch, args.num_envs)
        summary["unique_trajectory_count"] = float(len(sampled_trajectory_indices))
        print_metrics("random_contact_summary", summary)
        if args.require_terrain_context:
            validate_contact_metrics(summary, args.contact_tolerance, args.contact_normal_tolerance)
            print("[strict-context] PASS")
        return

    dataset_raw = load_dataset_batch(
        args.dataset,
        trajectory_start=args.trajectory_start,
        num_envs=args.num_envs,
        step=args.step,
        device=str(solver.torch_device),
        history_length=history_length,
    )
    metadata = {
        "trajectory_index": np.arange(args.trajectory_start, args.trajectory_start + args.num_envs),
        "step": np.full(args.num_envs, args.step),
    }
    metrics = compare_diagnostic_batch(
        env,
        args,
        dataset_raw,
        metadata,
        print_all_metrics=True,
        run_predictions=True,
    )
    if args.require_terrain_context:
        validate_contact_metrics(metrics, args.contact_tolerance, args.contact_normal_tolerance)
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
