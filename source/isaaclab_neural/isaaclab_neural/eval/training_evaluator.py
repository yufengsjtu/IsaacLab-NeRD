# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop rollout evaluator used by NeRD trainers."""

from __future__ import annotations

import logging

import numpy as np
import torch
import warp as wp
from isaaclab_neural.contacts.contact_set_schema import (
    is_contact_token_representation,
    is_native15_contact_representation,
)
from isaaclab_neural.data import TrajectoryDataset
from isaaclab_neural.eval.contact_set_matching import (
    contact_set_matching_metrics,
    contact_set_tolerance_failures,
    match_contact_tokens_by_identity,
)
from isaaclab_neural.utils.commons import JOINT_F_LIM, JOINT_Q_MAX, JOINT_Q_MIN, JOINT_QD_LIM
from newton import JointType
from torch.utils.data import default_collate
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrainingRolloutEvaluator:
    """Evaluate a trained NeRD model by rolling it from dataset states."""

    def __init__(
        self,
        neural_env,
        hdf5_dataset_path: str | None = None,
        eval_horizon: int = 10,
        device="cuda:0",
        require_terrain_context: bool = False,
        contact_context_validation: str | None = None,
        state_context_tolerance: float = 1.0e-5,
        contact_context_tolerance: float = 1.0e-4,
        contact_context_normal_tolerance: float = 1.0e-3,
        contact_context_velocity_tolerance: float = 1.0e-3,
    ):
        self.neural_env = neural_env
        self.device = device
        self.eval_horizon = eval_horizon
        self.history_length = int(getattr(neural_env.solver_neural, "num_states_history", 1))
        self.require_terrain_context = require_terrain_context
        if contact_context_validation is None:
            contact_context_validation = (
                "warn" if getattr(neural_env.solver_neural, "contact_mode", None) == "newton_native" else "strict"
            )
        if contact_context_validation not in ("strict", "warn"):
            raise ValueError(
                f"contact_context_validation must be 'strict' or 'warn', got {contact_context_validation!r}."
            )
        self.contact_context_validation = contact_context_validation
        self.state_context_tolerance = state_context_tolerance
        self.contact_context_tolerance = contact_context_tolerance
        self.contact_context_normal_tolerance = contact_context_normal_tolerance
        self.contact_context_velocity_tolerance = contact_context_velocity_tolerance
        self.trajectory_dataset = None
        if hdf5_dataset_path is not None:
            self.trajectory_dataset = TrajectoryDataset(
                hdf5_dataset_path=hdf5_dataset_path,
                sample_sequence_length=eval_horizon + self.history_length - 1,
                expected_contact_representation=getattr(neural_env.solver_neural, "contact_representation", None),
            )

    @torch.no_grad()
    def evaluate_action_mode(
        self,
        num_traj: int = 10,
        eval_mode: str = "rollout",
        trajectory_source: str = "dataset",
        eval_trajectories: dict[str, torch.Tensor] | None = None,
        zero_actions: bool = False,
        render: bool = False,
    ):
        """Evaluate action-driven rollouts."""
        trajectories = self._prepare_control_trajectories(
            control_key="actions",
            num_traj=num_traj,
            trajectory_source=trajectory_source,
            eval_trajectories=eval_trajectories,
            passive=zero_actions,
        )
        return self._evaluate_control_rollout(
            trajectories,
            control_key="actions",
            step_fn=self._step_action,
            eval_mode=eval_mode,
            render=render,
        )

    @torch.no_grad()
    def evaluate_joint_f_mode(
        self,
        num_traj: int = 10,
        eval_mode: str = "rollout",
        trajectory_source: str = "dataset",
        eval_trajectories: dict[str, torch.Tensor] | None = None,
        passive: bool = False,
        render: bool = False,
    ):
        """Evaluate direct joint-force rollouts."""
        trajectories = self._prepare_control_trajectories(
            control_key="joint_f",
            num_traj=num_traj,
            trajectory_source=trajectory_source,
            eval_trajectories=eval_trajectories,
            passive=passive,
        )
        return self._evaluate_control_rollout(
            trajectories,
            control_key="joint_f",
            step_fn=self._step_joint_f,
            eval_mode=eval_mode,
            render=render,
        )

    def _prepare_control_trajectories(
        self,
        *,
        control_key: str,
        num_traj: int,
        trajectory_source: str,
        eval_trajectories: dict[str, torch.Tensor] | None,
        passive: bool,
    ) -> dict[str, torch.Tensor]:
        if trajectory_source == "dataset":
            trajectories = self._sample_dataset_trajectories(num_traj)
        elif trajectory_source == "reference":
            if eval_trajectories is None:
                raise ValueError("eval_trajectories is required when trajectory_source='reference'.")
            trajectories = {
                "states": eval_trajectories["states"][:, :-1],
                "next_states": eval_trajectories["states"][:, 1:],
                control_key: eval_trajectories[control_key],
            }
        elif trajectory_source in ("sample", "env"):
            trajectories = self._sample_online_trajectories(
                control_key=control_key,
                num_traj=num_traj,
                initial_states_source=trajectory_source,
                passive=passive,
            )
        else:
            raise ValueError(f"Unsupported trajectory_source: {trajectory_source}")

        for key in ("states", "next_states", control_key):
            if key not in trajectories:
                raise KeyError(f"'{key}' is required for {control_key} eval with source '{trajectory_source}'.")
        return trajectories

    def _sample_dataset_trajectories(self, num_traj: int) -> dict[str, torch.Tensor]:
        if self.trajectory_dataset is None:
            raise ValueError("trajectory_source='dataset' requires an eval dataset path.")
        dataset_length = len(self.trajectory_dataset)
        if dataset_length == 0:
            raise ValueError("Eval dataset has no windows for the configured rollout horizon.")
        if num_traj > 0:
            indices = np.random.randint(low=0, high=dataset_length, size=num_traj)
        else:
            indices = np.arange(dataset_length)
        trajectories = default_collate([self.trajectory_dataset[int(index)] for index in indices])
        window_locations = self.trajectory_dataset.mapping_index2traj[indices]
        trajectories["_dataset_trajectory_index"] = torch.as_tensor(window_locations[:, 0], dtype=torch.long)
        trajectories["_dataset_window_start"] = torch.as_tensor(window_locations[:, 1], dtype=torch.long)
        return trajectories

    def _sample_online_trajectories(
        self,
        *,
        control_key: str,
        num_traj: int,
        initial_states_source: str,
        passive: bool,
    ) -> dict[str, torch.Tensor]:
        env = self.neural_env
        num_envs = env.num_envs
        total_traj = max(num_envs, ((num_traj + num_envs - 1) // num_envs) * num_envs)
        controls_dim = (
            env.unwrapped.action_space.shape[-1] if control_key == "actions" else env.solver_neural.joint_f_dim
        )
        states = torch.empty((total_traj, self.eval_horizon, env.solver_neural.state_dim), device=self.device)
        next_states = torch.empty_like(states)
        controls = torch.empty((total_traj, self.eval_horizon, controls_dim), device=self.device)

        for round_start in range(0, total_traj, num_envs):
            initial_states = None
            if initial_states_source == "sample":
                initial_states = self._sample_initial_states(num_envs)
            env.neural_adapter.reset(initial_states)
            for step in range(self.eval_horizon):
                states[round_start : round_start + num_envs, step].copy_(env.neural_adapter.states_torch)
                control = self._sample_control(control_key, passive=passive)
                controls[round_start : round_start + num_envs, step].copy_(control)
                step_fn = self._step_action if control_key == "actions" else self._step_joint_f
                next_states[round_start : round_start + num_envs, step].copy_(step_fn(control))
        return {"states": states[:num_traj], "next_states": next_states[:num_traj], control_key: controls[:num_traj]}

    def _sample_initial_states(self, num_envs: int) -> torch.Tensor:
        robot_name = self.neural_env.robot_name
        solver = self.neural_env.solver_neural
        dof_q = solver.dof_q_per_env
        dof_qd = solver.dof_qd_per_env
        q_min = self._limit_tensor(JOINT_Q_MIN.get(robot_name), dof_q, -1.0e6)
        q_max = self._limit_tensor(JOINT_Q_MAX.get(robot_name), dof_q, 1.0e6)
        qd_lim = self._limit_tensor(JOINT_QD_LIM.get(robot_name), dof_qd, 1.0e6)
        lower = torch.cat([q_min, -qd_lim])
        upper = torch.cat([q_max, qd_lim])
        samples = torch.rand((num_envs, solver.state_dim), device=self.device)
        return samples * (upper - lower) + lower

    def _limit_tensor(self, value, dim: int, default: float) -> torch.Tensor:
        if value is None:
            return torch.full((dim,), default, device=self.device, dtype=torch.float32)
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32).flatten()
        if tensor.numel() != dim:
            if tensor.numel() == 1:
                return torch.full((dim,), float(tensor.item()), device=self.device, dtype=torch.float32)
            raise ValueError(f"Expected limit with {dim} entries, got {tensor.numel()}.")
        return tensor

    def _sample_control(self, control_key: str, *, passive: bool) -> torch.Tensor:
        num_envs = self.neural_env.num_envs
        if control_key == "actions":
            action_shape = self.neural_env.unwrapped.action_space.shape
            action_dim = action_shape[-1]
            if passive:
                return torch.zeros((num_envs, action_dim), device=self.device)
            low = torch.as_tensor(
                self.neural_env.unwrapped.action_space.low, device=self.device, dtype=torch.float32
            ).flatten()
            high = torch.as_tensor(
                self.neural_env.unwrapped.action_space.high, device=self.device, dtype=torch.float32
            ).flatten()
            low = low[-action_dim:]
            high = high[-action_dim:]
            action = torch.rand((num_envs, action_dim), device=self.device)
            return action * (high - low) + low

        joint_f_dim = self.neural_env.solver_neural.joint_f_dim
        if passive:
            return torch.zeros((num_envs, joint_f_dim), device=self.device)
        limit = self._limit_tensor(JOINT_F_LIM.get(self.neural_env.robot_name), joint_f_dim, 1.0e6)
        joint_f = torch.rand((num_envs, joint_f_dim), device=self.device)
        return joint_f * (2.0 * limit) - limit

    def _evaluate_control_rollout(
        self,
        trajectories: dict[str, torch.Tensor],
        *,
        control_key: str,
        step_fn,
        eval_mode: str,
        render: bool,
    ):
        if eval_mode not in ("rollout", "single-step"):
            raise ValueError(f"Unsupported eval_mode: {eval_mode}")
        env = self.neural_env
        num_envs = env.num_envs
        total_traj = (trajectories["states"].shape[0] // num_envs) * num_envs
        if total_traj == 0:
            raise ValueError(f"num_traj must be at least num_envs ({num_envs}) for rollout eval.")

        history_offset = self._history_offset(trajectories)
        initial_states = trajectories["states"][:total_traj, history_offset, :].to(self.device)
        controls = trajectories[control_key][:total_traj, history_offset : history_offset + self.eval_horizon].to(
            self.device
        )
        target_next_states = trajectories["next_states"][
            :total_traj, history_offset : history_offset + self.eval_horizon
        ].to(self.device)

        rollout_states = torch.empty(
            (total_traj, self.eval_horizon + 1, env.solver_neural.state_dim),
            device=self.device,
        )
        if hasattr(env.solver_neural, "eval"):
            env.solver_neural.eval()

        num_rounds = total_traj // num_envs
        for round_index in tqdm(range(num_rounds)):
            start = round_index * num_envs
            end = start + num_envs
            rollout_states[start:end, 0].copy_(initial_states[start:end])
            self._restore_terrain_context(trajectories, start, end)
            self._reset_from_context(trajectories, start, end, history_offset)
            self._validate_runtime_state_context(trajectories, start, end, history_offset)
            self._preload_history(trajectories, start, end, history_offset)
            if eval_mode != "single-step":
                self._validate_runtime_contact_context(trajectories, start, end, history_offset)

            for step in range(self.eval_horizon):
                if eval_mode == "single-step":
                    state_step = history_offset + step
                    self._reset_from_context(trajectories, start, end, state_step)
                    self._validate_runtime_state_context(trajectories, start, end, state_step)
                    self._preload_history(trajectories, start, end, state_step)
                    self._validate_runtime_contact_context(trajectories, start, end, state_step)
                rollout_states[start:end, step + 1].copy_(step_fn(controls[start:end, step]))
                if render and hasattr(env.unwrapped, "render"):
                    env.unwrapped.render()

        next_states_diff, error_stats = self.calculate_error_metrics(target_next_states, rollout_states[:, 1:])
        trajectories_out = {
            "states": rollout_states,
            control_key: controls,
            "ground-truth": target_next_states,
        }
        return next_states_diff, trajectories_out, error_stats

    def _reset_from_context(
        self,
        trajectories: dict[str, torch.Tensor],
        start: int,
        end: int,
        step: int,
    ) -> None:
        """Reset generalized and maximal root state from one recorded frame."""
        root_body_q = trajectories.get("root_body_q")
        root_body_qd = trajectories.get("root_body_qd")
        if self.require_terrain_context and (root_body_q is None or root_body_qd is None):
            missing = [
                key for key, value in (("root_body_q", root_body_q), ("root_body_qd", root_body_qd)) if value is None
            ]
            raise ValueError(f"Strict rollout evaluation is missing root context fields: {missing}.")
        self.neural_env.neural_adapter.reset(
            trajectories["states"][start:end, step].to(self.device),
        )

    def _history_offset(self, trajectories: dict[str, torch.Tensor]) -> int:
        """Return the number of dataset frames used to warm-start history."""
        if self.history_length <= 1:
            return 0
        required = {"root_body_q", "states", "joint_f", "gravity_dir"}
        required.update(self.neural_env.solver_neural.contacts.keys())
        if not required.issubset(trajectories):
            if self.require_terrain_context:
                missing = sorted(required - set(trajectories))
                raise ValueError(f"Strict rollout evaluation is missing history inputs: {missing}.")
            return 0
        return self.history_length - 1

    def _preload_history(
        self,
        trajectories: dict[str, torch.Tensor],
        start: int,
        end: int,
        history_offset: int,
    ) -> None:
        """Preload raw dataset frames preceding the rollout state."""
        if history_offset == 0:
            return
        solver = self.neural_env.solver_neural
        preload = getattr(solver, "preload_states_history", None)
        if preload is None:
            if self.require_terrain_context:
                raise ValueError("Strict rollout evaluation requires solver history preloading support.")
            return
        keys = {"root_body_q", "states", "joint_f", "gravity_dir", *solver.contacts.keys()}
        missing = sorted(keys - set(trajectories))
        if missing:
            raise ValueError(f"Rollout history preload is missing dataset fields: {missing}.")
        history_start = history_offset - self.history_length + 1
        if history_start < 0:
            raise ValueError(
                f"Rollout history offset {history_offset} cannot preload {self.history_length - 1} prior frames."
            )
        preload({key: trajectories[key][start:end, history_start:history_offset].to(self.device) for key in keys})

    def _restore_terrain_context(
        self,
        trajectories: dict[str, torch.Tensor],
        start: int,
        end: int,
    ) -> None:
        """Map evaluation slots to the trajectory's recorded terrain patch.

        Flat/grid terrains only expose ``env_origins``. Curriculum terrains also
        expose ``terrain_levels`` / ``terrain_types``. Restore only the
        attributes that exist so flat eval does not fail when datasets still
        store sentinel ``terrain_level`` / ``terrain_type`` values.
        """
        keys = {"terrain_level", "terrain_type", "env_origin"}
        if not keys.issubset(trajectories):
            if self.require_terrain_context:
                raise ValueError("Strict rollout evaluation requires per-trajectory terrain patch metadata.")
            return
        terrain = getattr(getattr(self.neural_env.neural_adapter.isaaclab_env, "scene", None), "terrain", None)
        if terrain is None:
            if self.require_terrain_context:
                raise ValueError("Strict rollout evaluation requires an environment terrain.")
            return

        has_curriculum_terrain = (
            getattr(terrain, "terrain_levels", None) is not None and getattr(terrain, "terrain_types", None) is not None
        )
        if self.require_terrain_context and not has_curriculum_terrain:
            raise ValueError(
                "Strict rollout evaluation requires curriculum terrain attributes "
                "(terrain_levels and terrain_types), but the active TerrainImporter "
                "only provides a flat/grid layout."
            )
        if has_curriculum_terrain:
            terrain.terrain_levels[: end - start].copy_(trajectories["terrain_level"][start:end].to(terrain.device))
            terrain.terrain_types[: end - start].copy_(trajectories["terrain_type"][start:end].to(terrain.device))
        if getattr(terrain, "env_origins", None) is not None:
            terrain.env_origins[: end - start].copy_(trajectories["env_origin"][start:end].to(terrain.device))

    def _validate_runtime_state_context(
        self,
        trajectories: dict[str, torch.Tensor],
        start: int,
        end: int,
        history_offset: int,
    ) -> None:
        """Ensure reset-time state and root pose match the recorded frame."""
        if not self.require_terrain_context:
            return
        required = {"states", "root_body_q", "root_body_qd"}
        missing = sorted(required - set(trajectories))
        if missing:
            raise ValueError(f"Strict rollout evaluation is missing reset context fields: {missing}.")

        solver = self.neural_env.solver_neural
        runtime_context = {
            "states": solver.states,
            "root_body_q": solver.root_body_q,
            "root_body_qd": solver.root_body_qd,
        }
        failures = []
        for key, runtime_value in runtime_context.items():
            recorded = trajectories[key][start:end, history_offset].to(runtime_value.device)
            difference = torch.abs(runtime_value - recorded)
            if key == "root_body_q":
                quaternion_error = torch.minimum(
                    torch.abs(runtime_value[..., 3:7] - recorded[..., 3:7]),
                    torch.abs(runtime_value[..., 3:7] + recorded[..., 3:7]),
                )
                difference = torch.cat((difference[..., :3], quaternion_error), dim=-1)
            elif key == "states" and getattr(solver, "base_joint_type", None) == JointType.FREE:
                quaternion_error = torch.minimum(
                    torch.abs(runtime_value[..., 3:7] - recorded[..., 3:7]),
                    torch.abs(runtime_value[..., 3:7] + recorded[..., 3:7]),
                )
                difference = torch.cat(
                    (difference[..., :3], quaternion_error, difference[..., 7:]),
                    dim=-1,
                )
            max_error = torch.max(difference).item()
            if not np.isfinite(max_error) or max_error > self.state_context_tolerance:
                failures.append(f"{key}_max_error={max_error:.6g}")
        if failures:
            raise ValueError(
                "Runtime reset context mismatch: "
                + "; ".join(failures)
                + f" (tolerance={self.state_context_tolerance:.6g})."
            )

    def _validate_runtime_contact_context(
        self,
        trajectories: dict[str, torch.Tensor],
        start: int,
        end: int,
        history_offset: int,
    ) -> None:
        """Ensure reset-time runtime contacts match the recorded current frame."""
        if not self.require_terrain_context:
            return
        solver = self.neural_env.solver_neural
        if is_contact_token_representation(getattr(solver, "contact_representation", "flat")):
            self._validate_runtime_contact_tokens(trajectories, start, end, history_offset)
            return
        required_keys = {
            "contact_masks",
            "contact_points_0",
            "contact_points_1",
            "contact_normals",
            "contact_depths",
            "contact_thicknesses_0",
            "contact_thicknesses_1",
        }
        missing = sorted(required_keys - set(solver.contacts))
        if missing:
            raise ValueError(f"Runtime contact context is missing fields required for set matching: {missing}.")
        missing = sorted(set(solver.contacts) - set(trajectories))
        if missing:
            raise ValueError(f"Dataset contact context is missing runtime contact fields: {missing}.")

        dataset_contacts = {}
        runtime_contacts = {}
        slot_max_errors = {}
        for key, runtime_value in solver.contacts.items():
            recorded = trajectories[key][start:end, history_offset].to(runtime_value.device)
            dataset_contacts[key] = recorded.bool() if runtime_value.dtype == torch.bool else recorded
            runtime_contacts[key] = runtime_value
            if runtime_value.dtype != torch.bool:
                if not torch.isfinite(recorded).all() or not torch.isfinite(runtime_value).all():
                    raise ValueError(f"Runtime contact context for {key!r} contains non-finite values.")
                max_error = torch.max(torch.abs(runtime_value - recorded)).item()
                slot_max_errors[key] = max_error

        dataset_mask = dataset_contacts["contact_masks"]
        runtime_mask = runtime_contacts["contact_masks"]
        mask_mismatch = dataset_mask != runtime_mask
        normal_slot_error = slot_max_errors["contact_normals"]
        other_slot_errors = [error for key, error in slot_max_errors.items() if key != "contact_normals"]
        slots_match = (
            not mask_mismatch.any()
            and np.isfinite(normal_slot_error)
            and normal_slot_error <= self.contact_context_normal_tolerance
            and all(np.isfinite(error) and error <= self.contact_context_tolerance for error in other_slot_errors)
        )
        if slots_match:
            return

        num_contacts = dataset_mask.shape[-1]
        set_metrics = contact_set_matching_metrics(
            dataset_contacts,
            runtime_contacts,
            num_contacts,
            deduplication_tolerance=self.contact_context_tolerance,
            normal_tolerance=self.contact_context_normal_tolerance,
        )
        failures = contact_set_tolerance_failures(
            set_metrics,
            point_tolerance=self.contact_context_tolerance,
            normal_tolerance=self.contact_context_normal_tolerance,
        )

        suspect_envs = mask_mismatch.any(dim=-1)
        if not suspect_envs.any():
            dataset_points = dataset_contacts["contact_points_1"].reshape(end - start, num_contacts, 3)
            runtime_points = runtime_contacts["contact_points_1"].reshape(end - start, num_contacts, 3)
            point_slot_error = torch.linalg.vector_norm(dataset_points - runtime_points, dim=-1)
            suspect_envs = (point_slot_error > self.contact_context_tolerance).any(dim=-1)
        detail_rows = []
        for local_env in torch.nonzero(suspect_envs, as_tuple=False).flatten()[:8].tolist():
            slot_indices = torch.nonzero(mask_mismatch[local_env], as_tuple=False).flatten().tolist()
            batch_index = start + local_env
            trajectory_index = int(trajectories.get("_dataset_trajectory_index", torch.full((end,), -1))[batch_index])
            window_start = int(trajectories.get("_dataset_window_start", torch.full((end,), -1))[batch_index])
            terrain_level = int(trajectories.get("terrain_level", torch.full((end,), -1))[batch_index])
            terrain_type = int(trajectories.get("terrain_type", torch.full((end,), -1))[batch_index])
            detail_rows.append(
                f"trajectory={trajectory_index}, step={window_start + history_offset}, "
                f"terrain=({terrain_level}, {terrain_type}), mask_slots={slot_indices[:8]}, "
                f"dataset_active={int(dataset_mask[local_env].sum())}, "
                f"runtime_active={int(runtime_mask[local_env].sum())}"
            )

        mask_mismatch_count = int(mask_mismatch.sum().item())
        summary = (
            f"mask_mismatches={mask_mismatch_count}/{mask_mismatch.numel()}, "
            f"slot_point1_max={slot_max_errors['contact_points_1']:.6g}, "
            f"hausdorff_max={set_metrics['set_hausdorff_distance_max']:.6g}, "
            f"matched_normal_max={set_metrics['matched_normal_l2_max']:.6g}, "
            f"matched_depth_max={set_metrics['matched_depth_abs_max']:.6g}, "
            f"unique_count_diff_max={set_metrics['unique_count_abs_diff_max']:.6g}; " + " | ".join(detail_rows)
        )
        if failures:
            message = "Runtime contact-set geometry mismatch: " + "; ".join(failures) + f". Diagnostics: {summary}"
            if self.contact_context_validation == "strict":
                raise ValueError(message)
            logger.warning("%s Contact validation is diagnostic-only.", message)
            return
        logger.warning("Runtime contact slots differed, but unordered contact sets matched: %s", summary)

    def _validate_runtime_contact_tokens(
        self,
        trajectories: dict[str, torch.Tensor],
        start: int,
        end: int,
        history_offset: int,
    ) -> None:
        """Compare recorded and runtime contact-token sets for diagnostics."""
        solver = self.neural_env.solver_neural
        required = {"contact_tokens", "contact_token_overflow"}
        missing_runtime = sorted(required - set(solver.contacts))
        if missing_runtime:
            raise ValueError(f"Runtime contact-token context is missing fields: {missing_runtime}.")
        missing_dataset = sorted(required - set(trajectories))
        if missing_dataset:
            raise ValueError(f"Dataset contact-token context is missing fields: {missing_dataset}.")

        runtime_tokens = solver.contacts["contact_tokens"]
        recorded_tokens = trajectories["contact_tokens"][start:end, history_offset].to(runtime_tokens.device)
        if recorded_tokens.shape != runtime_tokens.shape:
            raise ValueError(
                f"Contact-token shape mismatch: dataset={tuple(recorded_tokens.shape)}, "
                f"runtime={tuple(runtime_tokens.shape)}."
            )
        if not torch.isfinite(recorded_tokens).all() or not torch.isfinite(runtime_tokens).all():
            raise ValueError("Runtime contact-token context contains non-finite values.")

        dataset_valid = recorded_tokens[..., 0] > 0.5
        runtime_valid = runtime_tokens[..., 0] > 0.5
        is_native15 = is_native15_contact_representation(getattr(solver, "contact_representation", "flat"))
        if is_native15:
            matches = match_contact_tokens_by_identity(
                recorded_tokens,
                runtime_tokens,
                identity_slice=slice(1, 2),
                point_slice=slice(2, 5),
                continuous_slice=slice(5, None),
            )
        else:
            matches = match_contact_tokens_by_identity(recorded_tokens, runtime_tokens)
        difference = matches.dataset - matches.runtime
        if is_native15:
            point_error = torch.linalg.vector_norm(difference[:, 2:5], dim=-1)
            secondary_point_error = torch.linalg.vector_norm(difference[:, 5:8], dim=-1)
            normal_error = torch.linalg.vector_norm(difference[:, 8:11], dim=-1)
            gap_error = difference[:, 11].abs()
            velocity_error = torch.linalg.vector_norm(difference[:, 12:15], dim=-1)
            margin_error = difference[:, 15:17].abs().amax(dim=-1)
        else:
            point_error = torch.linalg.vector_norm(difference[:, 4:7], dim=-1)
            secondary_point_error = torch.linalg.vector_norm(difference[:, 10:13], dim=-1)
            normal_error = torch.linalg.vector_norm(difference[:, 7:10], dim=-1)
            gap_error = difference[:, 13].abs()
            velocity_error = torch.linalg.vector_norm(difference[:, 14:17], dim=-1)
            margin_error = gap_error.new_zeros(gap_error.shape)
        geometry_mismatch = (
            (point_error > self.contact_context_tolerance)
            | (secondary_point_error > self.contact_context_tolerance)
            | (normal_error > self.contact_context_normal_tolerance)
            | (gap_error > self.contact_context_tolerance)
            | (velocity_error > self.contact_context_velocity_tolerance)
            | (margin_error > self.contact_context_tolerance)
        )

        def active_max(error: torch.Tensor) -> float:
            return float(error.max()) if error.numel() else 0.0

        runtime_overflow = solver.contacts["contact_token_overflow"]
        dataset_overflow = trajectories["contact_token_overflow"][start:end, history_offset].to(runtime_overflow.device)
        overflow_mismatch = dataset_overflow != runtime_overflow
        geometry_mismatch_per_env = torch.zeros(end - start, dtype=torch.bool, device=runtime_tokens.device)
        if geometry_mismatch.any():
            geometry_mismatch_per_env[matches.env_ids[geometry_mismatch]] = True
        tokens_match = (
            not matches.valid_mismatches_per_env.any()
            and not matches.categorical_mismatches_per_env.any()
            and not overflow_mismatch.any()
            and not geometry_mismatch.any()
        )
        if tokens_match:
            return

        suspect_envs = (
            (matches.valid_mismatches_per_env > 0)
            | (matches.categorical_mismatches_per_env > 0)
            | geometry_mismatch_per_env
            | overflow_mismatch
        )
        detail_rows = []
        for local_env in torch.nonzero(suspect_envs, as_tuple=False).flatten()[:8].tolist():
            batch_index = start + local_env
            trajectory_index = int(trajectories.get("_dataset_trajectory_index", torch.full((end,), -1))[batch_index])
            window_start = int(trajectories.get("_dataset_window_start", torch.full((end,), -1))[batch_index])
            terrain_level = int(trajectories.get("terrain_level", torch.full((end,), -1))[batch_index])
            terrain_type = int(trajectories.get("terrain_type", torch.full((end,), -1))[batch_index])
            detail_rows.append(
                f"trajectory={trajectory_index}, step={window_start + history_offset}, "
                f"terrain=({terrain_level}, {terrain_type}), "
                f"dataset_active={int(dataset_valid[local_env].sum())}, "
                f"runtime_active={int(runtime_valid[local_env].sum())}, "
                f"dataset_overflow={int(dataset_overflow[local_env])}, "
                f"runtime_overflow={int(runtime_overflow[local_env])}"
            )

        message = (
            "Runtime contact-token context mismatch: "
            f"valid_mismatches={int(matches.valid_mismatches_per_env.sum())}/{dataset_valid.numel()}, "
            f"categorical_mismatches={int(matches.categorical_mismatches_per_env.sum())}, "
            f"point_l2_max={active_max(point_error):.6g}, "
            f"normal_l2_max={active_max(normal_error):.6g}, "
            f"secondary_point_l2_max={active_max(secondary_point_error):.6g}, "
            f"gap_abs_max={active_max(gap_error):.6g}, "
            f"relative_velocity_l2_max={active_max(velocity_error):.6g}, "
            f"margin_abs_max={active_max(margin_error):.6g}, "
            f"overflow_mismatches={int(overflow_mismatch.sum())}; " + " | ".join(detail_rows)
        )
        if self.contact_context_validation == "strict":
            raise ValueError(message)
        logger.warning("%s Contact validation is diagnostic-only.", message)

    def _step_action(self, action: torch.Tensor) -> torch.Tensor:
        env = self.neural_env.unwrapped
        action = action.to(device=self.device, dtype=torch.float32)
        env.step(action)
        # The solver step already appends the pre-step state to sequence history.
        self.neural_env.neural_adapter.sync(update_history=False)
        return self.neural_env.neural_adapter.states_torch.detach().clone()

    def _step_joint_f(self, joint_f: torch.Tensor) -> torch.Tensor:
        adapter = self.neural_env.neural_adapter
        manager = adapter.manager
        joint_f = joint_f.to(device=self.device, dtype=torch.float32)
        manager._control.joint_f.assign(wp.from_torch(joint_f.reshape(-1)))
        manager._simulate_physics_only()
        env = adapter.isaaclab_env
        if hasattr(env, "scene"):
            env.scene.update(dt=getattr(env, "physics_dt", 0.0))
        # The solver step already appends the pre-step state to sequence history.
        adapter.sync(update_history=False)
        return adapter.states_torch.detach().clone()

    def calculate_error_metrics(self, target_next_states: torch.Tensor, rollout_states: torch.Tensor):
        solver = self.neural_env.solver_neural
        next_states_diff = target_next_states - rollout_states
        solver.wrap2PI(next_states_diff)

        error_stats = {"overall": {}, "step-wise": {}, "final": {}}
        mse_per_step = (next_states_diff**2).mean((0, 2))
        q_mse_per_step = (next_states_diff[..., : solver.dof_q_per_env] ** 2).mean((0, 2))
        qd_mse_per_step = (next_states_diff[..., solver.dof_q_per_env :] ** 2).mean((0, 2))
        l2_per_step = next_states_diff.norm(dim=-1).mean(0)

        error_stats["overall"]["error(MSE)"] = mse_per_step.mean()
        error_stats["overall"]["q_error(MSE)"] = q_mse_per_step.mean()
        error_stats["overall"]["qd_error(MSE)"] = qd_mse_per_step.mean()
        error_stats["overall"]["error(L2)"] = l2_per_step.mean()
        error_stats["step-wise"]["error(MSE)"] = mse_per_step
        error_stats["step-wise"]["q_error(MSE)"] = q_mse_per_step
        error_stats["step-wise"]["qd_error(MSE)"] = qd_mse_per_step
        error_stats["step-wise"]["error(L2)"] = l2_per_step
        error_stats["final"]["error(MSE)"] = mse_per_step[-1]
        error_stats["final"]["q_error(MSE)"] = q_mse_per_step[-1]
        error_stats["final"]["qd_error(MSE)"] = qd_mse_per_step[-1]
        error_stats["final"]["error(L2)"] = l2_per_step[-1]
        return next_states_diff, error_stats
