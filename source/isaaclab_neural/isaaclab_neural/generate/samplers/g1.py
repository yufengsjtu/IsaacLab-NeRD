# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 dataset trajectory sampler."""

from __future__ import annotations

from typing import Any

import torch

from isaaclab_neural.generate.adapter import DataGenerationAdapter
from isaaclab_neural.generate.sampler import ActionTrajectorySampler
from isaaclab_neural.robots.g1 import g1_hand_joint_indices
from isaaclab_neural.utils.commons import G1_BASE_Q_MAX, G1_BASE_Q_MIN, G1_BASE_QD_LIM


class G1TrajectorySampler(ActionTrajectorySampler):
    """G1 sampler with robot-derived limits, relative PD randomization, and held hands.

    Action-mode collection keeps stock numbered hand joints at the trajectory
    reset pose so random / pool-perturb rollouts do not flail the 14 finger DoF.
    Policy-mode collection is unchanged.
    """

    def __init__(
        self,
        adapter: DataGenerationAdapter,
        *,
        randomize_pd_gains: bool = False,
        kp_scale_range: tuple[float, float] = (0.8, 1.2),
        kd_scale_range: tuple[float, float] = (0.8, 1.2),
        hold_hand_actions: bool = True,
        **kwargs: Any,
    ):
        self._robot = self._resolve_robot(adapter)
        self._fill_robot_limits(kwargs)
        super().__init__(adapter, **kwargs)
        self.randomize_pd_gains = randomize_pd_gains
        self.kp_scale_range = kp_scale_range
        self.kd_scale_range = kd_scale_range
        self.hold_hand_actions = hold_hand_actions
        self._nominal_actuator_gains = self._capture_nominal_actuator_gains()
        self._held_hand_action: torch.Tensor | None = None
        self._hand_action_ids: torch.Tensor | None = None

    @staticmethod
    def _resolve_robot(adapter: DataGenerationAdapter):
        scene = getattr(adapter.env, "scene", None)
        if scene is None:
            return None
        try:
            return scene["robot"]
        except Exception:
            return getattr(scene, "articulations", {}).get("robot")

    @staticmethod
    def _torch_array(value) -> torch.Tensor:
        return getattr(value, "torch", value)

    def _fill_robot_limits(self, kwargs: dict) -> None:
        """Fill unspecified sampling limits from the initialized G1 articulation."""
        if self._robot is None:
            return

        data = self._robot.data
        if kwargs.get("joint_q_min") is None or kwargs.get("joint_q_max") is None:
            joint_pos_limits = self._torch_array(data.soft_joint_pos_limits)[0]
            if kwargs.get("joint_q_min") is None:
                kwargs["joint_q_min"] = torch.cat(
                    [
                        torch.tensor(G1_BASE_Q_MIN, device=joint_pos_limits.device, dtype=joint_pos_limits.dtype),
                        joint_pos_limits[:, 0],
                    ]
                )
            if kwargs.get("joint_q_max") is None:
                kwargs["joint_q_max"] = torch.cat(
                    [
                        torch.tensor(G1_BASE_Q_MAX, device=joint_pos_limits.device, dtype=joint_pos_limits.dtype),
                        joint_pos_limits[:, 1],
                    ]
                )

        if kwargs.get("joint_qd_lim") is None:
            joint_vel_limits = self._torch_array(data.soft_joint_vel_limits)[0]
            kwargs["joint_qd_lim"] = torch.cat(
                [
                    torch.tensor(G1_BASE_QD_LIM, device=joint_vel_limits.device, dtype=joint_vel_limits.dtype),
                    joint_vel_limits,
                ]
            )

        if kwargs.get("joint_f_lim") is None:
            joint_effort_limits = self._torch_array(data.joint_effort_limits)[0]
            kwargs["joint_f_lim"] = torch.cat(
                [
                    torch.zeros(6, device=joint_effort_limits.device, dtype=joint_effort_limits.dtype),
                    joint_effort_limits,
                ]
            )

    def _capture_nominal_actuator_gains(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        actuators = getattr(self._robot, "actuators", None)
        if not actuators:
            if self.randomize_pd_gains:
                raise RuntimeError("G1 PD-gain randomization requires initialized articulation actuators.")
            return {}

        return {
            name: (actuator.stiffness[0].detach().clone(), actuator.damping[0].detach().clone())
            for name, actuator in actuators.items()
        }

    def _joint_position_action_term(self):
        manager = getattr(self.adapter.env, "action_manager", None)
        if manager is None or not hasattr(manager, "get_term"):
            return None
        try:
            return manager.get_term("joint_pos")
        except (KeyError, AttributeError):
            return None

    def _on_action_rollout_reset(self, *, constrain_sampled_actions: bool = True) -> None:
        self._held_hand_action = None
        self._hand_action_ids = None
        if not self.hold_hand_actions or not constrain_sampled_actions or self._robot is None:
            return

        term = self._joint_position_action_term()
        if term is None:
            return
        joint_names = list(getattr(term, "_joint_names", ()) or ())
        hand_indices = g1_hand_joint_indices(joint_names)
        if not hand_indices:
            return

        joint_ids = getattr(term, "_joint_ids", slice(None))
        hand_ids = torch.as_tensor(hand_indices, device=self.adapter.device, dtype=torch.long)
        joint_pos = self._torch_array(self._robot.data.joint_pos)
        if isinstance(joint_ids, slice):
            current = joint_pos[:, hand_ids]
        else:
            articulation_ids = torch.as_tensor(joint_ids, device=joint_pos.device, dtype=torch.long)[hand_ids]
            current = joint_pos[:, articulation_ids]

        offset = getattr(term, "_offset", 0.0)
        scale = getattr(term, "_scale", 1.0)
        if isinstance(offset, torch.Tensor):
            offset = offset.to(device=current.device, dtype=current.dtype)[:, hand_ids]
        if isinstance(scale, torch.Tensor):
            scale = scale.to(device=current.device, dtype=current.dtype)[:, hand_ids]
        self._hand_action_ids = hand_ids
        self._held_hand_action = (current - offset) / scale

    def _constrain_sampled_action(self, action: torch.Tensor) -> torch.Tensor:
        if self._held_hand_action is None or self._hand_action_ids is None:
            return action
        action[:, self._hand_action_ids] = self._held_hand_action.to(device=action.device, dtype=action.dtype)
        return action

    def _pre_step_hook(self, step: int) -> None:
        if not self.randomize_pd_gains or step != 0:
            return

        env_ids = torch.arange(self.adapter.num_envs, device=self.adapter.device, dtype=torch.long)
        for name, actuator in self._robot.actuators.items():
            nominal_kp, nominal_kd = self._nominal_actuator_gains[name]
            kp_scale = torch.empty((self.adapter.num_envs, 1), device=self.adapter.device).uniform_(
                *self.kp_scale_range
            )
            kd_scale = torch.empty((self.adapter.num_envs, 1), device=self.adapter.device).uniform_(
                *self.kd_scale_range
            )
            stiffness = nominal_kp.unsqueeze(0) * kp_scale
            damping = nominal_kd.unsqueeze(0) * kd_scale
            actuator.stiffness[env_ids] = stiffness
            actuator.damping[env_ids] = damping

            # Pass actuator indices through; ``slice(None)`` means every joint.
            joint_ids = actuator.joint_indices
            if isinstance(joint_ids, torch.Tensor):
                joint_ids = joint_ids.to(device=self.adapter.device, dtype=torch.long)
            self._robot.write_joint_stiffness_to_sim_index(
                stiffness=stiffness,
                env_ids=env_ids,
                joint_ids=joint_ids,
            )
            self._robot.write_joint_damping_to_sim_index(
                damping=damping,
                env_ids=env_ids,
                joint_ids=joint_ids,
            )
