# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for G1 29-DoF data collection on the Newton backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def newton_free_joint_dof_starts(num_envs: int, device: str | torch.device) -> torch.Tensor:
    """Return the first generalized-force index of each environment's floating-base joint.

    Args:
        num_envs: Number of environments; each must contain exactly one floating-base articulation.
        device: Device of the returned tensor.

    Returns:
        Indices into Newton ``control.joint_f``, shape ``[num_envs]``.

    Raises:
        ValueError: The model does not contain exactly one free joint per environment.
    """
    import newton  # noqa: PLC0415
    from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

    model = NewtonManager.get_model()
    joint_type = model.joint_type.numpy()
    qd_start = model.joint_qd_start.numpy()
    starts = [int(qd_start[j]) for j in range(len(joint_type)) if joint_type[j] == int(newton.JointType.FREE)]
    if len(starts) != num_envs:
        raise ValueError(f"Expected one free joint per environment, found {len(starts)} for {num_envs} environments.")
    return torch.tensor(starts, dtype=torch.long, device=device)


def write_root_wrench(joint_f: torch.Tensor, dof_starts: torch.Tensor, wrench: torch.Tensor) -> None:
    """Write world-frame floating-base wrenches into Newton generalized forces.

    Newton applies the first three free-joint entries as a world-frame force and the last three as a world-frame
    torque on the root body. Implicit actuators only write joint entries, so the wrench persists across physics
    steps until it is overwritten.

    Args:
        joint_f: Flat generalized-force buffer (``control.joint_f`` as a torch view).
        dof_starts: Free-joint start indices, shape ``[N]``.
        wrench: Force [N] and torque [N·m] in the world frame, shape ``[N, 6]``.
    """
    joint_f[dof_starts[:, None] + torch.arange(6, device=dof_starts.device)] = wrench


class apply_root_wrench_pulses(ManagerTermBase):
    """Apply random horizontal force pulses at the floating base through Newton's free-joint generalized force.

    Pushes enter NeRD through the recorded ``joint_f`` root entries instead of an unmodelled velocity jump. Use the
    term in ``"interval"`` mode with ``interval_range_s`` equal to the environment step so that it updates every step;
    pulse onsets, magnitudes and directions are sampled per environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        import warp as wp  # noqa: PLC0415
        from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

        self._dof_starts = newton_free_joint_dof_starts(env.num_envs, env.device)
        self._joint_f = wp.to_torch(NewtonManager.get_control().joint_f)
        self._wrench = torch.zeros(env.num_envs, 6, device=env.device)
        self._remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._wait = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._resample_wait(slice(None), cfg.params.get("wait_range_s", (2.0, 5.0)), env.step_dt)

    def _resample_wait(self, env_ids, wait_range_s: tuple[float, float], step_dt: float) -> None:
        count = self._wait[env_ids].numel()
        low, high = (max(1, round(value / step_dt)) for value in wait_range_s)
        self._wait[env_ids] = torch.randint(low, high + 1, (count,), device=self._wait.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        force_range: tuple[float, float] = (0.0, 400.0),
        torque_range: tuple[float, float] = (0.0, 0.0),
        duration_s: float = 0.1,
        wait_range_s: tuple[float, float] = (2.0, 5.0),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Advance pulse timers and write the active wrenches.

        Args:
            env: Environment.
            env_ids: Environments selected by the event manager; all environments are advanced.
            force_range: Horizontal force magnitude range [N].
            torque_range: Torque magnitude range about a random axis [N·m].
            duration_s: Pulse duration [s].
            wait_range_s: Range of the interval between pulse onsets [s].
            asset_cfg: Floating-base robot.
        """
        del env_ids, asset_cfg
        self._wait -= 1
        start = (self._wait <= 0) & (self._remaining == 0)
        count = int(start.sum())
        if count:
            device = self._wrench.device
            yaw = torch.rand(count, device=device) * 2.0 * math.pi
            force = math_utils.sample_uniform(*force_range, (count,), device=device)
            self._wrench[start] = 0.0
            self._wrench[start, 0] = force * torch.cos(yaw)
            self._wrench[start, 1] = force * torch.sin(yaw)
            axis = torch.nn.functional.normalize(torch.randn(count, 3, device=device), dim=-1)
            self._wrench[start, 3:] = axis * math_utils.sample_uniform(*torque_range, (count, 1), device=device)
            self._remaining[start] = max(1, round(duration_s / env.step_dt))
            self._resample_wait(start, wait_range_s, env.step_dt)
        active = self._remaining > 0
        write_root_wrench(self._joint_f, self._dof_starts, self._wrench * active[:, None])
        self._remaining[active] -= 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._remaining[env_ids] = 0
        self._wrench[env_ids] = 0.0
        write_root_wrench(self._joint_f, self._dof_starts, self._wrench)


def reset_root_lying(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    fraction: float,
    height: float = 0.25,
    joint_noise: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset a random fraction of environments to lie on the back, front or a side with random yaw.

    Apply this term after the regular root and joint reset terms; the selected environments override their state.

    Args:
        env: Environment.
        env_ids: Environments being reset.
        fraction: Probability that a reset environment starts lying down.
        height: Initial root height above the environment origin [m].
        joint_noise: Standard deviation of the joint-position noise around the defaults [rad].
        asset_cfg: Robot to reset.
    """
    asset = env.scene[asset_cfg.name]
    env_ids = env_ids[torch.rand(len(env_ids), device=env_ids.device) < fraction]
    if len(env_ids) == 0:
        return
    count = len(env_ids)
    device = asset.device
    kind = torch.randint(0, 4, (count,), device=device)
    half_pi = math.pi / 2
    roll = torch.where(kind == 2, half_pi, torch.where(kind == 3, -half_pi, 0.0))
    pitch = torch.where(kind == 0, -half_pi, torch.where(kind == 1, half_pi, 0.0))
    yaw = (torch.rand(count, device=device) * 2.0 - 1.0) * math.pi
    pose = torch.zeros(count, 7, device=device)
    pose[:, :3] = env.scene.env_origins[env_ids]
    pose[:, 2] += height
    pose[:, 3:] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(count, 6, device=device), env_ids=env_ids)
    limits = asset.data.joint_pos_limits.torch[env_ids]
    joint_pos = asset.data.default_joint_pos.torch[env_ids] + joint_noise * torch.randn(
        count, asset.num_joints, device=device
    )
    joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
