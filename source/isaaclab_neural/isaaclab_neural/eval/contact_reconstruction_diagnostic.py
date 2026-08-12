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
from isaaclab_neural.data.datasets import validate_contact_token_metadata
from isaaclab_neural.eval.contact_set_matching import (
    contact_set_matching_metrics,
    contact_set_tolerance_failures,
    match_contact_tokens_by_identity,
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
        help="Require terrain provenance and restore and validate trajectory terrain patches.",
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
    parser.add_argument(
        "--contact-velocity-tolerance",
        type=float,
        default=1.0e-3,
        help="Maximum accepted contact-token relative-velocity L2 error.",
    )
    parser.add_argument(
        "--contact-context-validation",
        choices=("strict", "warn"),
        default=None,
        help="Contact mismatch policy. Defaults to warn for Newton-native contacts.",
    )
    parser.add_argument("--num-contacts-per-env", type=int, default=64)
    parser.add_argument("--contact-mode", choices=("fixed_ground", "newton_native"), default="newton_native")
    parser.add_argument(
        "--contact-packing-policy",
        choices=(
            "stable_index",
            "penetration_priority",
            "random",
            "force_priority",
            "body_round_robin_pair_atomic",
        ),
        default="penetration_priority",
    )
    parser.add_argument(
        "--contact-representation",
        choices=("flat", "contact_tokens"),
        default="flat",
    )
    parser.add_argument("--max-contact-tokens", type=int, default=64)
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
            contact_representation=args.contact_representation,
            max_contact_tokens=args.max_contact_tokens,
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
            int(args.terrain_seed_override) if args.terrain_seed_override is not None else int(terrain_context["seed"])
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
        validate_contact_token_metadata(data_group)
        required = {"states", "next_states", "joint_f", "root_body_q", "root_body_qd", "gravity_dir"}
        if "contact_tokens" in data_group:
            required.update({"contact_tokens", "contact_token_overflow"})
        else:
            required.update(
                {
                    "contact_masks",
                    "contact_normals",
                    "contact_depths",
                    "contact_points_0",
                    "contact_points_1",
                    "contact_thicknesses_0",
                    "contact_thicknesses_1",
                }
            )
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
            traj_lengths = np.asarray(cast(h5py.Dataset, data_group["traj_lengths"])[trajectory_start:trajectory_end])
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
            elif np.issubdtype(array.dtype, np.integer):
                batch[key] = torch.as_tensor(array, dtype=torch.long, device=device)
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
        validate_contact_token_metadata(data_group)
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
                f"No trajectories are long enough for history_length={history_length} and eval_horizon={eval_horizon}."
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
            elif np.issubdtype(array.dtype, np.integer):
                batch[key] = torch.as_tensor(array, dtype=torch.long, device=device)
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


def contact_token_metrics(
    dataset_inputs: dict[str, torch.Tensor],
    runtime_inputs: dict[str, torch.Tensor],
    *,
    point_tolerance: float = 1.0e-4,
    normal_tolerance: float = 1.0e-3,
    velocity_tolerance: float = 1.0e-3,
) -> dict[str, float]:
    """Compare latest-frame tokens using identity-constrained geometric matches."""
    dataset_tokens = dataset_inputs["contact_tokens"][:, -1]
    runtime_tokens = runtime_inputs["contact_tokens"][:, -1]
    if dataset_tokens.shape != runtime_tokens.shape:
        raise ValueError(
            f"Contact-token shape mismatch: dataset={tuple(dataset_tokens.shape)}, "
            f"runtime={tuple(runtime_tokens.shape)}."
        )
    if not torch.isfinite(dataset_tokens).all() or not torch.isfinite(runtime_tokens).all():
        raise ValueError("Contact-token comparison contains non-finite values.")

    dataset_valid = dataset_tokens[..., 0] > 0.5
    runtime_valid = runtime_tokens[..., 0] > 0.5
    matches = match_contact_tokens_by_identity(dataset_tokens, runtime_tokens)
    difference = matches.dataset - matches.runtime
    point_error = torch.linalg.vector_norm(difference[:, 4:7], dim=-1)
    normal_error = torch.linalg.vector_norm(difference[:, 7:10], dim=-1)
    lever_error = torch.linalg.vector_norm(difference[:, 10:13], dim=-1)
    gap_error = difference[:, 13].abs()
    velocity_error = torch.linalg.vector_norm(difference[:, 14:17], dim=-1)
    dataset_overflow = dataset_inputs["contact_token_overflow"][:, -1]
    runtime_overflow = runtime_inputs["contact_token_overflow"][:, -1]

    geometry_mismatch = (
        (point_error > point_tolerance)
        | (normal_error > normal_tolerance)
        | (lever_error > point_tolerance)
        | (gap_error > point_tolerance)
        | (velocity_error > velocity_tolerance)
    )
    geometry_mismatch_per_env = torch.zeros(dataset_tokens.shape[0], dtype=torch.bool, device=dataset_tokens.device)
    if geometry_mismatch.any():
        geometry_mismatch_per_env[matches.env_ids[geometry_mismatch]] = True
    mismatched_env = (matches.valid_mismatches_per_env > 0) | (matches.categorical_mismatches_per_env > 0)
    mismatched_env |= geometry_mismatch_per_env | (dataset_overflow != runtime_overflow)
    metrics = {
        "valid_mismatch_count": float(matches.valid_mismatches_per_env.sum()),
        "valid_total_count": float(dataset_valid.numel()),
        "mismatched_env_count": float(mismatched_env.sum()),
        "dataset_active_mean": float(dataset_valid.sum(dim=-1).float().mean()),
        "runtime_active_mean": float(runtime_valid.sum(dim=-1).float().mean()),
        "categorical_mismatch_count": float(matches.categorical_mismatches_per_env.sum()),
        "shared_valid_count": float(matches.dataset.shape[0]),
        "overflow_mismatch_count": float((dataset_overflow != runtime_overflow).sum()),
    }
    for name, error in (
        ("point_l2", point_error),
        ("normal_l2", normal_error),
        ("lever_l2", lever_error),
        ("gap_abs", gap_error),
        ("relative_velocity_l2", velocity_error),
    ):
        metrics[f"{name}_sum"] = float(error.sum())
        metrics[f"{name}_mean"] = float(error.mean()) if error.numel() else 0.0
        metrics[f"{name}_max"] = float(error.max()) if error.numel() else 0.0
    return metrics


def root_replay_metrics(
    dataset_inputs: dict[str, torch.Tensor],
    runtime_inputs: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Measure reset-time generalized and maximal root-state reconstruction."""
    dataset_states = dataset_inputs["states"][:, -1]
    runtime_states = runtime_inputs["states"][:, -1]
    dataset_q = dataset_inputs["root_body_q"][:, -1]
    runtime_q = runtime_inputs["root_body_q"][:, -1]
    dataset_qd = dataset_inputs["root_body_qd"][:, -1]
    runtime_qd = runtime_inputs["root_body_qd"][:, -1]

    position_delta = runtime_q[:, :3] - dataset_q[:, :3]
    velocity_delta = runtime_qd - dataset_qd
    quaternion_error = torch.minimum(
        torch.linalg.vector_norm(runtime_q[:, 3:7] - dataset_q[:, 3:7], dim=-1),
        torch.linalg.vector_norm(runtime_q[:, 3:7] + dataset_q[:, 3:7], dim=-1),
    )
    return {
        "states_abs_max": float(torch.abs(runtime_states - dataset_states).max()),
        "root_position_l2_mean": float(torch.linalg.vector_norm(position_delta, dim=-1).mean()),
        "root_position_l2_max": float(torch.linalg.vector_norm(position_delta, dim=-1).max()),
        "root_quaternion_sign_invariant_l2_max": float(quaternion_error.max()),
        "root_linear_velocity_l2_max": float(torch.linalg.vector_norm(velocity_delta[:, :3], dim=-1).max()),
        "root_angular_velocity_l2_max": float(torch.linalg.vector_norm(velocity_delta[:, 3:6], dim=-1).max()),
        "root_position_delta_x_mean": float(position_delta[:, 0].mean()),
        "root_position_delta_y_mean": float(position_delta[:, 1].mean()),
        "root_position_delta_z_mean": float(position_delta[:, 2].mean()),
    }


def raw_token_frame_metrics(
    dataset_inputs: dict[str, torch.Tensor],
    runtime_inputs: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Relate world-space token-point errors to root-position replay errors."""
    matches = match_contact_tokens_by_identity(
        dataset_inputs["contact_tokens"][:, -1],
        runtime_inputs["contact_tokens"][:, -1],
    )
    if matches.dataset.shape[0] == 0:
        return {
            "shared_valid_count": 0.0,
            "world_point_delta_l2_mean": 0.0,
            "world_point_delta_l2_max": 0.0,
            "point_minus_root_delta_l2_mean": 0.0,
            "point_minus_root_delta_l2_max": 0.0,
        }

    point_delta = matches.runtime[:, 4:7] - matches.dataset[:, 4:7]
    root_delta = runtime_inputs["root_body_q"][:, -1, :3] - dataset_inputs["root_body_q"][:, -1, :3]
    residual = point_delta - root_delta[matches.env_ids]
    point_error = torch.linalg.vector_norm(point_delta, dim=-1)
    residual_error = torch.linalg.vector_norm(residual, dim=-1)
    return {
        "shared_valid_count": float(matches.dataset.shape[0]),
        "world_point_delta_l2_mean": float(point_error.mean()),
        "world_point_delta_l2_max": float(point_error.max()),
        "point_minus_root_delta_l2_mean": float(residual_error.mean()),
        "point_minus_root_delta_l2_max": float(residual_error.max()),
        "world_point_delta_x_mean": float(point_delta[:, 0].mean()),
        "world_point_delta_y_mean": float(point_delta[:, 1].mean()),
        "world_point_delta_z_mean": float(point_delta[:, 2].mean()),
    }


def analyze_dataset_token_row_ownership(dataset_path: str, body_world: torch.Tensor) -> None:
    """Validate recorded token owner ids against trajectory world-row metadata."""
    with h5py.File(Path(dataset_path).expanduser(), "r", swmr=True, libver="latest") as dataset_file:
        data_group = cast(h5py.Group, dataset_file["data"])
        if "contact_tokens" not in data_group:
            return
        validate_contact_token_metadata(data_group)
        tokens_dataset = cast(h5py.Dataset, data_group["contact_tokens"])
        required_data = {"contact_token_body_ids", "contact_token_world_ids"}
        missing_data = sorted(required_data - set(data_group.keys()))
        if missing_data:
            raise ValueError(f"Contact-token diagnostic requires identity fields: {missing_data}.")
        if "context" not in dataset_file or "trajectories" not in cast(h5py.Group, dataset_file["context"]):
            raise ValueError("Contact-token diagnostic requires per-trajectory world context.")
        trajectory_group = cast(h5py.Group, cast(h5py.Group, dataset_file["context"])["trajectories"])
        required_context = {"state_world_id", "root_world_id", "contact_world_id"}
        missing_context = sorted(required_context - set(trajectory_group.keys()))
        if missing_context:
            raise ValueError(f"Contact-token diagnostic requires world context fields: {missing_context}.")

        token_world_dataset = cast(h5py.Dataset, data_group["contact_token_world_ids"])
        token_body_dataset = cast(h5py.Dataset, data_group["contact_token_body_ids"])
        expected_world_ids = torch.as_tensor(
            np.asarray(cast(h5py.Dataset, trajectory_group["contact_world_id"])), dtype=torch.long
        )
        counts = {
            "active_token_count": 0,
            "recorded_token_world_row_mismatch_count": 0,
            "owner_body_world_mismatch_count": 0,
            "owner_body_runtime_comparable_count": 0,
            "owner_body_runtime_unavailable_count": 0,
            "invalid_token_world_id_count": 0,
            "missing_valid_owner_body_count": 0,
            "invalid_owner_body_count": 0,
        }
        mismatch_examples = []
        chunk_size = 64
        for chunk_start in range(0, tokens_dataset.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, tokens_dataset.shape[0])
            valid = torch.as_tensor(np.asarray(tokens_dataset[chunk_start:chunk_end, :, :, 0])) > 0.5
            token_world_ids = torch.as_tensor(np.asarray(token_world_dataset[chunk_start:chunk_end]), dtype=torch.long)
            token_body_ids = torch.as_tensor(np.asarray(token_body_dataset[chunk_start:chunk_end]), dtype=torch.long)
            expected = expected_world_ids[chunk_start:chunk_end, None, None].expand_as(token_world_ids)
            world_mismatch = (token_world_ids != expected) & valid
            missing_valid_owner = valid & (token_body_ids < 0)
            body_runtime_unavailable = valid & (token_body_ids >= body_world.shape[0])
            runtime_comparable = valid & ~missing_valid_owner & ~body_runtime_unavailable
            body_owner_world_mismatch = torch.zeros_like(valid)
            body_owner_world_mismatch[runtime_comparable] = (
                body_world[token_body_ids[runtime_comparable]].to(dtype=torch.long)
                != token_world_ids[runtime_comparable]
            )

            counts["active_token_count"] += int(valid.sum())
            counts["recorded_token_world_row_mismatch_count"] += int(world_mismatch.sum())
            counts["owner_body_world_mismatch_count"] += int(body_owner_world_mismatch.sum())
            counts["owner_body_runtime_comparable_count"] += int(runtime_comparable.sum())
            counts["owner_body_runtime_unavailable_count"] += int(body_runtime_unavailable.sum())
            counts["invalid_token_world_id_count"] += int(((token_world_ids >= 0) & ~valid).sum())
            counts["missing_valid_owner_body_count"] += int(missing_valid_owner.sum())
            counts["invalid_owner_body_count"] += int(((token_body_ids >= 0) & ~valid).sum())

            identity_mismatch = world_mismatch | body_owner_world_mismatch | missing_valid_owner
            if len(mismatch_examples) < 16 and identity_mismatch.any():
                for row, step, slot in torch.nonzero(identity_mismatch, as_tuple=False).tolist():
                    mismatch_examples.append(
                        (
                            chunk_start + row,
                            step,
                            slot,
                            int(token_body_ids[row, step, slot]),
                            int(token_world_ids[row, step, slot]),
                            int(expected[row, step, slot]),
                        )
                    )
                    if len(mismatch_examples) == 16:
                        break

        print_metrics("dataset_row_ownership", {name: float(value) for name, value in counts.items()})
        for row, step, slot, body_id, recorded_world_id, expected_world_id in mismatch_examples:
            print(
                "owner_world_mismatch="
                f"(trajectory={row}, step={step}, slot={slot}, body_id={body_id}, "
                f"recorded_world_id={recorded_world_id}, expected_world_id={expected_world_id})"
            )
        world_id_arrays = {
            name: np.asarray(cast(h5py.Dataset, trajectory_group[name]))
            for name in ("state_world_id", "root_world_id", "contact_world_id")
        }
        state_world = world_id_arrays["state_world_id"]
        state_root_mismatch_count = int(np.sum(state_world != world_id_arrays["root_world_id"]))
        state_contact_mismatch_count = int(np.sum(state_world != world_id_arrays["contact_world_id"]))
        print(f"state_root_world_mismatch_count={state_root_mismatch_count}")
        print(f"state_contact_world_mismatch_count={state_contact_mismatch_count}")

        failures = {
            name: count
            for name, count in (
                (
                    "recorded_token_world_row_mismatch_count",
                    counts["recorded_token_world_row_mismatch_count"],
                ),
                ("owner_body_world_mismatch_count", counts["owner_body_world_mismatch_count"]),
                ("missing_valid_owner_body_count", counts["missing_valid_owner_body_count"]),
                ("invalid_token_world_id_count", counts["invalid_token_world_id_count"]),
                ("invalid_owner_body_count", counts["invalid_owner_body_count"]),
                ("state_root_world_mismatch_count", state_root_mismatch_count),
                ("state_contact_world_mismatch_count", state_contact_mismatch_count),
            )
            if count
        }
        if failures:
            details = ", ".join(f"{name}={count}" for name, count in failures.items())
            raise ValueError(f"Contact-token ownership schema mismatch: {details}.")


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
    """Restore recorded terrain patch assignments for the selected trajectories.

    Flat/grid terrains only expose ``env_origins``. Curriculum terrains also
    expose ``terrain_levels`` / ``terrain_types``. Restore only attributes that
    exist so flat diagnostics do not fail on sentinel level/type metadata.
    """
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
    has_curriculum_terrain = (
        getattr(terrain, "terrain_levels", None) is not None and getattr(terrain, "terrain_types", None) is not None
    )
    if required and not has_curriculum_terrain:
        raise ValueError(
            "Strict terrain restoration requires curriculum terrain attributes "
            "(terrain_levels and terrain_types), but the active TerrainImporter "
            "only provides a flat/grid layout."
        )
    if has_curriculum_terrain:
        terrain.terrain_levels[:num_envs].copy_(batch["terrain_level"].to(terrain.device, dtype=torch.long))
        terrain.terrain_types[:num_envs].copy_(batch["terrain_type"].to(terrain.device, dtype=torch.long))
    if getattr(terrain, "env_origins", None) is not None:
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


def validate_contact_token_metrics(
    metrics: dict[str, float],
    point_tolerance: float,
    normal_tolerance: float = 1.0e-3,
    velocity_tolerance: float = 1.0e-3,
) -> None:
    """Fail when reconstructed contact-token slots differ materially."""
    failures = []
    if metrics["valid_mismatch_count"] != 0:
        failures.append(f"valid_mismatch_count={metrics['valid_mismatch_count']:.8g}")
    if metrics["categorical_mismatch_count"] != 0:
        failures.append(f"categorical_mismatch_count={metrics['categorical_mismatch_count']:.8g}")
    if metrics["overflow_mismatch_count"] != 0:
        failures.append(f"overflow_mismatch_count={metrics['overflow_mismatch_count']:.8g}")
    for key, tolerance in (
        ("point_l2_max", point_tolerance),
        ("normal_l2_max", normal_tolerance),
        ("lever_l2_max", point_tolerance),
        ("gap_abs_max", point_tolerance),
        ("relative_velocity_l2_max", velocity_tolerance),
    ):
        value = metrics[key]
        if not np.isfinite(value) or value > tolerance:
            failures.append(f"{key}={value:.8g} exceeds {tolerance:.8g}")
    if failures:
        raise ValueError("Contact-token reconstruction mismatch: " + "; ".join(failures) + ".")


def validate_or_warn_contact_metrics(
    metrics: dict[str, float],
    args: argparse.Namespace,
    *,
    is_contact_tokens: bool,
) -> bool:
    """Apply strict or diagnostic-only contact validation."""
    validation = args.contact_context_validation
    if validation is None:
        validation = "warn" if args.contact_mode == "newton_native" else "strict"
    try:
        if is_contact_tokens:
            validate_contact_token_metrics(
                metrics,
                args.contact_tolerance,
                args.contact_normal_tolerance,
                args.contact_velocity_tolerance,
            )
        else:
            validate_contact_metrics(metrics, args.contact_tolerance, args.contact_normal_tolerance)
    except ValueError as error:
        if validation == "strict":
            raise
        print(f"[contact-context-warning] {error} Contact validation is diagnostic-only.")
        return False
    return True


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

    is_contact_tokens = "contact_tokens" in dataset_raw
    if is_contact_tokens:
        dataset_capacity = dataset_raw["contact_tokens"].shape[-2]
        runtime_capacity = int(getattr(solver, "max_contact_tokens", 0) or solver.num_contacts_per_env)
        if dataset_capacity != runtime_capacity:
            raise ValueError(
                f"Dataset contact-token capacity ({dataset_capacity}) does not match "
                f"runtime capacity ({runtime_capacity})."
            )
    elif dataset_raw["contact_masks"].shape[-1] != solver.num_contacts_per_env:
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
    if print_all_metrics:
        print_metrics("root_replay", root_replay_metrics(dataset_raw, runtime_raw))
        if is_contact_tokens:
            print_metrics(
                "raw_world_contact_comparison",
                contact_token_metrics(
                    dataset_raw,
                    runtime_raw,
                    point_tolerance=args.contact_tolerance,
                    normal_tolerance=args.contact_normal_tolerance,
                    velocity_tolerance=args.contact_velocity_tolerance,
                ),
            )
            print_metrics("raw_token_frame", raw_token_frame_metrics(dataset_raw, runtime_raw))

    comparison_metrics = (
        contact_token_metrics(
            dataset_processed,
            runtime_processed,
            point_tolerance=args.contact_tolerance,
            normal_tolerance=args.contact_normal_tolerance,
            velocity_tolerance=args.contact_velocity_tolerance,
        )
        if is_contact_tokens
        else contact_metrics(
            dataset_processed,
            runtime_processed,
            solver.num_contacts_per_env,
            deduplication_tolerance=args.contact_tolerance,
            normal_tolerance=args.contact_normal_tolerance,
        )
    )
    mismatch_count = comparison_metrics["mismatched_env_count"]
    if print_all_metrics or mismatch_count > 0:
        print_metrics("contact_comparison", comparison_metrics)
        if not is_contact_tokens:
            print_metrics(
                "input_differences",
                input_difference_metrics(dataset_processed, runtime_processed, solver.num_contacts_per_env),
            )
    if not is_contact_tokens and comparison_metrics["mask_mismatch_count"] > 0:
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
        [metrics["slot_point_distance_max"] for metrics in metrics_by_batch if metrics["shared_active_slot_count"] > 0],
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


def aggregate_contact_token_metrics(metrics_by_batch: list[dict[str, float]], batch_size: int) -> dict[str, float]:
    """Aggregate equal-sized contact-token diagnostic batches."""
    shared_valid_count = sum(metrics["shared_valid_count"] for metrics in metrics_by_batch)
    summary = {
        "valid_mismatch_count": sum(metrics["valid_mismatch_count"] for metrics in metrics_by_batch),
        "valid_total_count": sum(metrics["valid_total_count"] for metrics in metrics_by_batch),
        "mismatched_env_count": sum(metrics["mismatched_env_count"] for metrics in metrics_by_batch),
        "dataset_active_mean": float(np.mean([metrics["dataset_active_mean"] for metrics in metrics_by_batch])),
        "runtime_active_mean": float(np.mean([metrics["runtime_active_mean"] for metrics in metrics_by_batch])),
        "categorical_mismatch_count": sum(metrics["categorical_mismatch_count"] for metrics in metrics_by_batch),
        "shared_valid_count": shared_valid_count,
        "overflow_mismatch_count": sum(metrics["overflow_mismatch_count"] for metrics in metrics_by_batch),
        "num_samples": float(len(metrics_by_batch) * batch_size),
    }
    for name in ("point_l2", "normal_l2", "lever_l2", "gap_abs", "relative_velocity_l2"):
        total = sum(metrics[f"{name}_sum"] for metrics in metrics_by_batch)
        summary[f"{name}_sum"] = total
        summary[f"{name}_mean"] = total / shared_valid_count if shared_valid_count > 0 else 0.0
        summary[f"{name}_max"] = max(metrics[f"{name}_max"] for metrics in metrics_by_batch)
    return summary


def run_diagnostic(env, args: argparse.Namespace) -> None:
    """Run dataset-contact and runtime-contact one-step A/B predictions."""
    solver = env.neural_adapter.solver
    history_length = int(getattr(solver, "num_states_history", 1))
    is_contact_tokens = getattr(solver, "contact_representation", "flat") == "contact_tokens"
    print(
        "[setup] "
        f"dataset={args.dataset}, checkpoint={args.checkpoint or 'none'}, num_envs={args.num_envs}, "
        f"history_length={history_length}, random_samples={args.random_samples}"
    )
    if is_contact_tokens:
        contact_adapter = solver.contact_adapter
        if contact_adapter is None or contact_adapter.body_world is None:
            raise ValueError("Contact-token diagnostic requires Newton body_world metadata.")
        analyze_dataset_token_row_ownership(args.dataset, torch.as_tensor(contact_adapter.body_world, dtype=torch.long))
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
                print_all_metrics=batch_index == 0,
                run_predictions=False,
            )
            metrics_by_batch.append(metrics)
            print(
                f"[random-progress] batch={batch_index + 1}/{args.random_samples // args.num_envs}, "
                f"mismatched_envs={int(metrics['mismatched_env_count'])}"
            )
        summary = (
            aggregate_contact_token_metrics(metrics_by_batch, args.num_envs)
            if is_contact_tokens
            else aggregate_contact_metrics(metrics_by_batch, args.num_envs)
        )
        summary["unique_trajectory_count"] = float(len(sampled_trajectory_indices))
        print_metrics("random_contact_summary", summary)
        if args.require_terrain_context:
            if validate_or_warn_contact_metrics(summary, args, is_contact_tokens=is_contact_tokens):
                print("[contact-context] PASS")
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
        if validate_or_warn_contact_metrics(metrics, args, is_contact_tokens=is_contact_tokens):
            print("[contact-context] PASS")


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
