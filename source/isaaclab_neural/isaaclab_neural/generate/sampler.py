# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory sampling helpers for NeRD dataset generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM, is_contact_token_representation
from isaaclab_neural.generate.adapter import DataGenerationAdapter


class ActionTrajectorySampler:
    """Collect fixed-length action-mode trajectories for NeRD training."""

    def __init__(
        self,
        adapter: DataGenerationAdapter,
        *,
        trajectory_length: int,
        data_device: str | torch.device,
        action_low: float = -1.0,
        action_high: float = 1.0,
        joint_q_min: float | np.ndarray | torch.Tensor | None = None,
        joint_q_max: float | np.ndarray | torch.Tensor | None = None,
        joint_qd_lim: float | np.ndarray | torch.Tensor | None = None,
        joint_f_lim: float | np.ndarray | torch.Tensor | None = None,
    ):
        self.adapter = adapter
        self.trajectory_length = trajectory_length
        self.data_device = torch.device(data_device)
        self.action_low, self.action_high = adapter.action_bounds(low=action_low, high=action_high)
        self.states_min, self.states_max = adapter.state_bounds(
            joint_q_min=joint_q_min,
            joint_q_max=joint_q_max,
            joint_qd_lim=joint_qd_lim,
        )
        self.joint_f_lim = adapter.joint_force_bounds(joint_f_lim)

    def allocate_batch_buffers(self, record_actions: bool = True) -> dict[str, Any]:
        """Allocate per-rollout batch buffers."""
        num_envs = self.adapter.num_envs
        trajectory_length = self.trajectory_length
        num_contacts_per_env = self.adapter.num_contacts_per_env
        use_contact_tokens = is_contact_token_representation(
            getattr(self.adapter.solver, "contact_representation", "flat")
        )
        buffers: dict[str, Any] = {
            "states": torch.empty((num_envs, trajectory_length, self.adapter.state_dim), device=self.data_device),
            "next_states": torch.empty((num_envs, trajectory_length, self.adapter.state_dim), device=self.data_device),
            "joint_f": torch.empty((num_envs, trajectory_length, self.adapter.joint_f_dim), device=self.data_device),
            "root_body_q": torch.empty((num_envs, trajectory_length, 7), device=self.data_device),
            "root_body_qd": torch.empty((num_envs, trajectory_length, 6), device=self.data_device),
            "gravity_dir": torch.zeros((num_envs, trajectory_length, 3), device=self.data_device),
        }
        if use_contact_tokens:
            max_tokens = int(getattr(self.adapter.solver, "max_contact_tokens", num_contacts_per_env))
            buffers["contacts"] = {
                "contact_tokens": torch.empty(
                    (num_envs, trajectory_length, max_tokens, CONTACT_TOKEN_DIM),
                    device=self.data_device,
                ),
                "contact_token_overflow": torch.empty(
                    (num_envs, trajectory_length),
                    dtype=torch.long,
                    device=self.data_device,
                ),
            }
            buffers["contact_token_body_ids"] = torch.empty(
                (num_envs, trajectory_length, max_tokens),
                dtype=torch.long,
                device=self.data_device,
            )
            buffers["contact_token_world_ids"] = torch.empty(
                (num_envs, trajectory_length, max_tokens),
                dtype=torch.long,
                device=self.data_device,
            )
        else:
            buffers["contacts"] = {
                "contact_masks": torch.empty(
                    (num_envs, trajectory_length, num_contacts_per_env),
                    dtype=torch.bool,
                    device=self.data_device,
                ),
                "contact_normals": torch.empty(
                    (num_envs, trajectory_length, num_contacts_per_env * 3), device=self.data_device
                ),
                "contact_depths": torch.empty(
                    (num_envs, trajectory_length, num_contacts_per_env), device=self.data_device
                ),
                "contact_points_0": torch.empty(
                    (num_envs, trajectory_length, num_contacts_per_env * 3), device=self.data_device
                ),
                "contact_points_1": torch.empty(
                    (num_envs, trajectory_length, num_contacts_per_env * 3), device=self.data_device
                ),
                "contact_thicknesses_0": torch.empty(
                    (num_envs, trajectory_length, num_contacts_per_env), device=self.data_device
                ),
                "contact_thicknesses_1": torch.empty(
                    (num_envs, trajectory_length, num_contacts_per_env), device=self.data_device
                ),
            }
        buffers["trajectory_context"] = self._trajectory_context()
        if record_actions:
            buffers["actions"] = torch.empty(
                (num_envs, trajectory_length, self.adapter.action_dim), device=self.data_device
            )
        up_axis = int(getattr(self.adapter.model, "up_axis", 2))
        buffers["gravity_dir"][:, :, up_axis] = -1.0
        return buffers

    def _trajectory_context(self) -> dict[str, torch.Tensor]:
        """Capture the source environment and terrain patch for each trajectory."""
        num_envs = self.adapter.num_envs
        source_env_id = torch.arange(num_envs, dtype=torch.int64, device=self.data_device)
        terrain_level = torch.full((num_envs,), -1, dtype=torch.int64, device=self.data_device)
        terrain_type = torch.full((num_envs,), -1, dtype=torch.int64, device=self.data_device)
        env_origin = torch.zeros((num_envs, 3), dtype=torch.float32, device=self.data_device)

        scene = getattr(self.adapter.env, "scene", None)
        terrain = getattr(scene, "terrain", None)
        if terrain is not None:
            if getattr(terrain, "terrain_levels", None) is not None:
                terrain_level.copy_(terrain.terrain_levels.to(self.data_device))
            if getattr(terrain, "terrain_types", None) is not None:
                terrain_type.copy_(terrain.terrain_types.to(self.data_device))
            if getattr(terrain, "env_origins", None) is not None:
                env_origin.copy_(terrain.env_origins.to(self.data_device))

        state_world_ids = self.adapter.state_world_ids.to(self.data_device)
        root_world_ids = self.adapter.root_world_ids.to(self.data_device)
        contact_world_ids = self.adapter.contact_world_ids.to(self.data_device)
        if not torch.equal(state_world_ids, root_world_ids) or not torch.equal(state_world_ids, contact_world_ids):
            raise RuntimeError(
                "State, root, and contact rows must represent the same Newton worlds: "
                f"state={state_world_ids.tolist()}, root={root_world_ids.tolist()}, "
                f"contact={contact_world_ids.tolist()}."
            )
        return {
            "source_env_id": source_env_id,
            "state_world_id": state_world_ids,
            "root_world_id": root_world_ids,
            "contact_world_id": contact_world_ids,
            "terrain_level": terrain_level,
            "terrain_type": terrain_type,
            "env_origin": env_origin,
        }

    def _copy_before_step(self, buffers: dict[str, Any], step: int) -> None:
        inputs = self.adapter.raw_neural_inputs()
        self._copy_input(buffers["states"][:, step], inputs["states"])
        self._copy_input(buffers["root_body_q"][:, step], inputs["root_body_q"])
        self._copy_input(buffers["root_body_qd"][:, step], inputs["root_body_qd"])
        if "contact_token_body_ids" in buffers:
            body_ids = self.adapter.contact_token_body_ids
            if body_ids is None:
                raise RuntimeError("Contact-token generation is missing owner-body ids.")
            self._copy_input(buffers["contact_token_body_ids"][:, step], body_ids)
            token_valid = inputs["contact_tokens"][..., 0] > 0.5
            token_world_ids = self.adapter.contact_token_world_ids
            if token_world_ids is None:
                raise RuntimeError("Contact-token generation is missing owner-world ids.")
            expected_world_ids = self.adapter.contact_world_ids.unsqueeze(-1).expand_as(body_ids)
            mismatched = token_valid.squeeze(1) & (token_world_ids != expected_world_ids)
            if mismatched.any():
                rows, slots = torch.nonzero(mismatched, as_tuple=True)
                raise RuntimeError(
                    "Packed contact-token rows do not match owner Newton worlds: "
                    f"rows={rows[:8].tolist()}, slots={slots[:8].tolist()}, "
                    f"owners={token_world_ids[mismatched][:8].tolist()}."
                )
            self._copy_input(buffers["contact_token_world_ids"][:, step], token_world_ids)
        contacts = buffers["contacts"]
        for key in contacts:
            self._copy_input(contacts[key][:, step], inputs[key])

    def _copy_after_step(self, buffers: dict[str, Any], step: int, action: torch.Tensor) -> None:
        inputs = self.adapter.raw_neural_inputs()
        self._copy_input(buffers["next_states"][:, step], inputs["states"])
        self._copy_input(buffers["joint_f"][:, step], inputs["joint_f"])
        if "actions" in buffers:
            buffers["actions"][:, step].copy_(action.to(self.data_device))

    def _copy_input(self, destination: torch.Tensor, source: torch.Tensor) -> None:
        """Copy a solver input tensor into a per-step rollout buffer."""
        source = source.to(self.data_device)
        while source.ndim > destination.ndim and source.shape[1] == 1:
            source = source.squeeze(1)
        destination.copy_(source)

    @staticmethod
    def _append_valid_trajectories(batch: dict[str, Any], rollout_batches: dict[str, Any]) -> int:
        invalid_masks = (
            batch["next_states"].isnan().any(dim=(1, 2))
            | batch["next_states"].isinf().any(dim=(1, 2))
            | (batch["next_states"] > 1e5).any(dim=(1, 2))
            | (batch["next_states"] < -1e5).any(dim=(1, 2))
        )
        valid_masks = ~invalid_masks
        for key, value in batch.items():
            if key not in rollout_batches:
                rollout_batches[key] = {} if isinstance(value, dict) else []
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    rollout_batches[key].setdefault(sub_key, [])
                    rollout_batches[key][sub_key].append(sub_value[valid_masks].clone())
            else:
                rollout_batches[key].append(value[valid_masks].clone())
        return int(valid_masks.sum().item() * batch["next_states"].shape[1])

    @staticmethod
    def _merge_rollout_batches(rollout_batches: dict[str, Any]) -> dict[str, Any]:
        """Concatenate collected rollout rounds.

        Large contact-token tensors are merged on CPU to avoid a second peak GPU
        allocation that dwarfs the per-round buffers.
        """
        rollouts: dict[str, Any] = {}
        for key, value in rollout_batches.items():
            if isinstance(value, dict):
                rollouts[key] = {
                    sub_key: torch.cat([tensor.detach().cpu() for tensor in sub_value], dim=0)
                    for sub_key, sub_value in value.items()
                }
            else:
                rollouts[key] = torch.cat([tensor.detach().cpu() for tensor in value], dim=0)
        return rollouts

    def _resolve_initial_states(
        self,
        initial_states_source: str,
        initial_states_pool: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if initial_states_source == "env":
            return None
        if initial_states_source == "sample":
            return self.adapter.sample_initial_states(self.states_min, self.states_max)
        if initial_states_source == "input_states_pool":
            if initial_states_pool is None:
                raise ValueError("initial_states_pool is required when initial_states_source='input_states_pool'.")
            if initial_states_pool.ndim != 2:
                raise ValueError("initial_states_pool must be a 2D tensor.")
            indices = torch.randint(
                low=0,
                high=initial_states_pool.shape[0],
                size=(self.adapter.num_envs,),
                device=self.adapter.device,
            )
            return initial_states_pool.to(self.adapter.device)[indices]
        raise ValueError(f"Unsupported initial_states_source: {initial_states_source}")

    def _sample_loop(
        self,
        num_transitions: int,
        *,
        record_actions: bool,
        initial_states_source: str,
        initial_states_pool: torch.Tensor | None,
        render: bool,
        step_fn,
        pre_rollout_fn=None,
    ) -> dict[str, Any]:
        rollout_batches: dict[str, Any] = {}
        total_transitions = 0
        progress_bar = tqdm(total=num_transitions)

        while total_transitions < num_transitions:
            initial_states = self._resolve_initial_states(initial_states_source, initial_states_pool)
            self.adapter.reset(initial_states=initial_states)
            if pre_rollout_fn is not None:
                pre_rollout_fn()
            batch = self.allocate_batch_buffers(record_actions=record_actions)

            for step in range(self.trajectory_length):
                self._copy_before_step(batch, step)
                self._pre_step_hook(step)
                control = step_fn(step)
                self._copy_after_step(
                    batch, step, control if record_actions else torch.empty(0, device=self.adapter.device)
                )
                if render and hasattr(self.adapter.env, "render"):
                    self.adapter.env.render()

            valid_transitions = self._append_valid_trajectories(batch, rollout_batches)
            del batch
            if valid_transitions == 0:
                raise RuntimeError(
                    "Dataset generation produced no valid transitions for a full rollout batch. "
                    "Check the sampling ranges, initial states, and contact configuration."
                )
            total_transitions += valid_transitions
            progress_bar.update(valid_transitions)

        progress_bar.close()
        print(f"\nTotal number of transitions generated: {total_transitions}")
        return self._merge_rollout_batches(rollout_batches)

    def _pre_step_hook(self, step: int) -> None:
        """Hook for environment-specific per-frame effects."""
        pass

    def sample_trajectories_action_mode(
        self,
        num_transitions: int,
        *,
        zero_actions: bool = False,
        initial_states_source: str = "sample",
        initial_states_pool: torch.Tensor | None = None,
        step_granularity: str = "frame",
        render: bool = False,
    ) -> dict[str, Any]:
        """Collect trajectories with sampled actions."""
        decimation = 1
        if step_granularity == "frame":
            decimation = max(1, int(getattr(getattr(self.adapter.env, "cfg", None), "decimation", 1)))
        held_action: torch.Tensor | None = None

        def reset_held_action() -> None:
            nonlocal held_action
            held_action = None

        def step_fn(step: int) -> torch.Tensor:
            nonlocal held_action
            should_process_action = held_action is None or step_granularity == "env" or step % decimation == 0
            if should_process_action:
                held_action = (
                    self.adapter.zero_action()
                    if zero_actions
                    else self.adapter.sample_action(self.action_low, self.action_high)
                )
            action = held_action
            if action is None:
                raise RuntimeError("Action sampling failed before stepping.")
            if step_granularity == "frame":
                self.adapter.step_action_frame(action, process_action=should_process_action)
            elif step_granularity == "env":
                self.adapter.step(action)
            else:
                raise ValueError(f"Unsupported step_granularity: {step_granularity}")
            return action

        return self._sample_loop(
            num_transitions,
            record_actions=True,
            initial_states_source=initial_states_source,
            initial_states_pool=initial_states_pool,
            render=render,
            step_fn=step_fn,
            pre_rollout_fn=reset_held_action,
        )

    def sample_trajectories_joint_f_mode(
        self,
        num_transitions: int,
        *,
        passive: bool = False,
        initial_states_source: str = "sample",
        initial_states_pool: torch.Tensor | None = None,
        render: bool = False,
    ) -> dict[str, Any]:
        """Collect trajectories with direct sampled joint forces."""

        def step_fn(_step: int) -> torch.Tensor:
            joint_f = torch.zeros((self.adapter.num_envs, self.adapter.joint_f_dim), device=self.adapter.device)
            if not passive and self.adapter.joint_f_dim > 0:
                joint_f = self.adapter.sample_joint_f(self.joint_f_lim)
            self.adapter.step_joint_f_frame(joint_f)
            return joint_f

        return self._sample_loop(
            num_transitions,
            record_actions=False,
            initial_states_source=initial_states_source,
            initial_states_pool=initial_states_pool,
            render=render,
            step_fn=step_fn,
        )

    def sample_trajectories_policy_mode(
        self,
        num_transitions: int,
        *,
        policy,
        env_wrapped=None,
        initial_states_source: str = "env",
        render: bool = False,
    ) -> dict[str, Any]:
        """Collect trajectories driven by a callable policy."""
        decimation = max(1, int(getattr(getattr(self.adapter.env, "cfg", None), "decimation", 1)))
        policy_has_reset = hasattr(policy, "reset")
        policy_action: torch.Tensor | None = None

        def reset_policy_state() -> None:
            nonlocal policy_action
            policy_action = None
            if policy_has_reset:
                dones = torch.ones(self.adapter.num_envs, dtype=torch.long, device=self.adapter.device)
                policy.reset(dones)

        def step_fn(step: int) -> torch.Tensor:
            nonlocal policy_action
            should_process_action = policy_action is None or step % decimation == 0
            if should_process_action:
                if env_wrapped is not None and hasattr(env_wrapped, "get_observations"):
                    obs = env_wrapped.get_observations()
                elif hasattr(self.adapter.env, "get_observations"):
                    obs = self.adapter.env.get_observations()
                else:
                    obs = self.adapter.env.obs_buf
                policy_action = policy(obs)
            if policy_action is None:
                raise RuntimeError("Policy did not produce an action before stepping.")
            self.adapter.step_action_frame(policy_action, process_action=should_process_action)
            return policy_action

        with torch.inference_mode():
            return self._sample_loop(
                num_transitions,
                record_actions=True,
                initial_states_source=initial_states_source,
                initial_states_pool=None,
                render=render,
                step_fn=step_fn,
                pre_rollout_fn=reset_policy_state,
            )

    def sample(self, num_transitions: int, *, zero_actions: bool = False, render: bool = False) -> dict[str, Any]:
        """Sample trajectories until at least ``num_transitions`` valid transitions are collected."""
        return self.sample_trajectories_action_mode(
            num_transitions,
            zero_actions=zero_actions,
            initial_states_source="env",
            step_granularity="env",
            render=render,
        )


TrajectorySampler = ActionTrajectorySampler
