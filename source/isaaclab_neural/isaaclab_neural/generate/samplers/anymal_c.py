# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anymal-C dataset trajectory sampler."""

from __future__ import annotations

import torch

from isaaclab_neural.generate.sampler import ActionTrajectorySampler


class AnymalCTrajectorySampler(ActionTrajectorySampler):
    """Anymal-C sampler with optional per-frame actuator gain randomization."""

    def __init__(
        self,
        *args,
        randomize_pd_gains: bool = False,
        kp_range: tuple[float, float] = (20.0, 80.0),
        kd_range: tuple[float, float] = (0.5, 4.0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.randomize_pd_gains = randomize_pd_gains
        self.kp_range = kp_range
        self.kd_range = kd_range
        self._robot = self._resolve_robot()

    def _resolve_robot(self):
        scene = getattr(self.adapter.env, "scene", None)
        if scene is None:
            return None
        try:
            return scene["robot"]
        except Exception:
            return getattr(scene, "articulations", {}).get("robot")

    def _pre_step_hook(self, step: int) -> None:
        if not self.randomize_pd_gains or self._robot is None:
            return
        env_ids = torch.arange(self.adapter.num_envs, device=self.adapter.device, dtype=torch.long)
        kp_per_env = torch.empty((self.adapter.num_envs, 1), device=self.adapter.device).uniform_(*self.kp_range)
        kd_per_env = torch.empty((self.adapter.num_envs, 1), device=self.adapter.device).uniform_(*self.kd_range)
        if self._write_actuator_gains(env_ids, kp_per_env, kd_per_env):
            return

        num_joints = int(getattr(self._robot, "num_joints", self.adapter.joint_f_dim))
        joint_ids = torch.arange(num_joints, device=self.adapter.device, dtype=torch.long)
        kp = kp_per_env.expand(self.adapter.num_envs, num_joints)
        kd = kd_per_env.expand(self.adapter.num_envs, num_joints)
        if hasattr(self._robot, "write_actuator_stiffness_to_sim"):
            self._robot.write_actuator_stiffness_to_sim(stiffness=kp, env_ids=env_ids, joint_ids=joint_ids)
        if hasattr(self._robot, "write_actuator_damping_to_sim"):
            self._robot.write_actuator_damping_to_sim(damping=kd, env_ids=env_ids, joint_ids=joint_ids)

    def _write_actuator_gains(self, env_ids: torch.Tensor, kp_per_env: torch.Tensor, kd_per_env: torch.Tensor) -> bool:
        """Write gains to explicit actuator tensors when available."""
        actuators = getattr(self._robot, "actuators", None)
        if not actuators:
            return False

        for actuator in actuators.values():
            kp = kp_per_env.expand(env_ids.shape[0], actuator.num_joints)
            kd = kd_per_env.expand(env_ids.shape[0], actuator.num_joints)
            actuator.stiffness[env_ids] = kp
            actuator.damping[env_ids] = kd
        return True
