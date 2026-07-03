# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop rollout evaluator used by NeRD trainers."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import default_collate
from tqdm import tqdm
import warp as wp

from isaaclab_neural.data import TrajectoryDataset
from isaaclab_neural.utils.commons import JOINT_F_LIM, JOINT_Q_MAX, JOINT_Q_MIN, JOINT_QD_LIM


class TrainingRolloutEvaluator:
    """Evaluate a trained NeRD model by rolling it from dataset states."""

    def __init__(self, neural_env, hdf5_dataset_path: str | None = None, eval_horizon: int = 10, device="cuda:0"):
        self.neural_env = neural_env
        self.device = device
        self.eval_horizon = eval_horizon
        self.trajectory_dataset = None
        if hdf5_dataset_path is not None:
            self.trajectory_dataset = TrajectoryDataset(
                hdf5_dataset_path=hdf5_dataset_path,
                sample_sequence_length=eval_horizon,
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
        return {key: trajectories[key] for key in ("states", "next_states", control_key)}

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
        return default_collate([self.trajectory_dataset[int(index)] for index in indices])

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
        controls_dim = env.unwrapped.action_space.shape[-1] if control_key == "actions" else env.solver_neural.joint_f_dim
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
            low = torch.as_tensor(self.neural_env.unwrapped.action_space.low, device=self.device, dtype=torch.float32).flatten()
            high = torch.as_tensor(self.neural_env.unwrapped.action_space.high, device=self.device, dtype=torch.float32).flatten()
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

        initial_states = trajectories["states"][:total_traj, 0, :].to(self.device)
        controls = trajectories[control_key][:total_traj].to(self.device)
        target_next_states = trajectories["next_states"][:total_traj].to(self.device)

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
            env.neural_adapter.reset(initial_states[start:end])

            for step in range(self.eval_horizon):
                if eval_mode == "single-step":
                    env.neural_adapter.reset(trajectories["states"][start:end, step].to(self.device))
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
