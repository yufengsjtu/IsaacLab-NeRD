# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event, command and reward terms for G1 29-DoF tasks on the Newton backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg, EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ..contacts.geometry_mdp import body_contacts

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


def body_roll_pitch(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the heading-frame roll and pitch of one body [rad].

    Args:
        env: Environment.
        asset_cfg: Robot with exactly one resolved body.

    Returns:
        Roll and pitch, each of shape ``[N]``.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids[0]]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(quat)
    return math_utils.wrap_to_pi(roll), math_utils.wrap_to_pi(pitch)


class BaseHeightTorsoCommand(CommandTerm):
    """Uniformly sampled targets for root height and torso roll and pitch.

    The command is ``[height, roll, pitch]``: root height above the environment origin [m] and heading-frame torso
    roll and pitch [rad].
    """

    cfg: BaseHeightTorsoCommandCfg

    def __init__(self, cfg: BaseHeightTorsoCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._env = env
        self._torso_cfg = SceneEntityCfg(cfg.asset_name, body_names=cfg.torso_body_name)
        self._torso_cfg.resolve(env.scene)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_torso"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Targets ``[height, roll, pitch]`` [m, rad, rad], shape ``[N, 3]``."""
        return self._command

    def _update_metrics(self) -> None:
        asset = self._env.scene[self.cfg.asset_name]
        height = asset.data.root_pos_w.torch[:, 2] - self._env.scene.env_origins[:, 2]
        roll, pitch = body_roll_pitch(self._env, self._torso_cfg)
        self.metrics["error_height"] = (height - self._command[:, 0]).abs()
        self.metrics["error_torso"] = torch.hypot(roll - self._command[:, 1], pitch - self._command[:, 2])

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        count = len(env_ids)
        for column, bounds in enumerate((self.cfg.ranges.height, self.cfg.ranges.roll, self.cfg.ranges.pitch)):
            self._command[env_ids, column] = math_utils.sample_uniform(*bounds, (count,), device=self.device)

    def _update_command(self) -> None:
        pass


@configclass
class BaseHeightTorsoCommandCfg(CommandTermCfg):
    """Configuration of :class:`BaseHeightTorsoCommand`."""

    class_type: type = BaseHeightTorsoCommand

    asset_name: str = "robot"
    torso_body_name: str = "torso_link"

    @configclass
    class Ranges:
        """Uniform sampling ranges."""

        height: tuple[float, float] = (0.45, 0.78)
        """Root height above the environment origin [m]."""

        roll: tuple[float, float] = (-0.3, 0.3)
        """Torso roll [rad]."""

        pitch: tuple[float, float] = (-0.3, 0.6)
        """Torso pitch [rad]; positive leans forward."""

    ranges: Ranges = Ranges()


def track_base_height_exp(
    env: ManagerBasedEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward root-height tracking with ``exp(-err^2 / std^2)``.

    Args:
        env: Environment.
        command_name: :class:`BaseHeightTorsoCommand` term name.
        std: Error scale [m].
        asset_cfg: Robot.

    Returns:
        Reward per environment.
    """
    height = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    error = height - env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-error.square() / std**2)


def track_torso_roll_pitch_exp(
    env: ManagerBasedEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward torso roll and pitch tracking with ``exp(-|err|^2 / std^2)``.

    Args:
        env: Environment.
        command_name: :class:`BaseHeightTorsoCommand` term name.
        std: Error scale [rad].
        asset_cfg: Robot with the torso body resolved.

    Returns:
        Reward per environment.
    """
    roll, pitch = body_roll_pitch(env, asset_cfg)
    command = env.command_manager.get_command(command_name)
    error = (roll - command[:, 1]).square() + (pitch - command[:, 2]).square()
    return torch.exp(-error / std**2)


# Walking-gait terms ported from the 29-DoF G1 velocity tasks of isaac-sim/IsaacLab PR #8055. Root height is measured
# from the environment origin, which equals the PR's terrain-scan estimate on flat ground.


def joint_deviation_l2(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the squared displacement of selected joints from their default positions [rad²].

    Args:
        env: Environment.
        asset_cfg: Robot with the selected joints resolved.

    Returns:
        Penalty per environment.
    """
    data = env.scene[asset_cfg.name].data
    joint_ids = asset_cfg.joint_ids
    return (data.joint_pos.torch[:, joint_ids] - data.default_joint_pos.torch[:, joint_ids]).square().sum(dim=1)


def feet_flight(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return one when all selected feet are airborne, otherwise zero.

    Args:
        env: Environment.
        sensor_cfg: Contact sensor with the feet resolved.

    Returns:
        Indicator per environment.
    """
    air_time = env.scene.sensors[sensor_cfg.name].data.current_air_time.torch[:, sensor_cfg.body_ids]
    return torch.all(air_time > 0.0, dim=-1).float()


def geometry_feet_flight(
    env: ManagerBasedEnv, body_cfg: SceneEntityCfg, maximum_separation: float = 0.0
) -> torch.Tensor:
    """Return one when no selected foot touches geometry, otherwise zero; :func:`feet_flight` for NeRD physics.

    Args:
        env: Environment with NeRD Newton-native contacts.
        body_cfg: Robot with the feet resolved.
        maximum_separation: Largest signed surface separation counted as contact [m].

    Returns:
        Indicator per environment.
    """
    contact = body_contacts(body_cfg.body_ids, maximum_separation=maximum_separation, env=env)
    return (~contact.any(dim=1)).float()


class feet_air_time_variance(ManagerTermBase):
    """Penalize the variance across feet of completed swing and stance durations [s²].

    Phases are measured at the control rate; initial partial phases are ignored and durations are clipped at
    ``max_time`` [s]. The penalty is active only while the commanded planar speed exceeds 0.1 m/s.
    """

    _feet_param = "sensor_cfg"
    """Term parameter whose ``body_ids`` select the feet."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        num_feet = len(cfg.params[self._feet_param].body_ids)
        self._elapsed = torch.zeros(env.num_envs, num_feet, device=env.device)
        self._air = torch.zeros_like(self._elapsed)
        self._stance = torch.zeros_like(self._elapsed)
        self._contact = torch.zeros_like(self._elapsed, dtype=torch.bool)
        self._known = torch.zeros_like(self._contact)
        self._initialized = torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        for buffer in (self._elapsed, self._air, self._stance, self._contact, self._known, self._initialized):
            buffer[env_ids] = 0

    def __call__(
        self, env: ManagerBasedEnv, command_name: str, sensor_cfg: SceneEntityCfg, max_time: float = 0.5
    ) -> torch.Tensor:
        """Update the phase timers and return the penalty.

        Args:
            env: Environment.
            command_name: Velocity command term name.
            sensor_cfg: Contact sensor with the feet resolved.
            max_time: Duration clip [s].

        Returns:
            Penalty per environment [s²].
        """
        contact = env.scene.sensors[sensor_cfg.name].data.current_air_time.torch[:, sensor_cfg.body_ids] <= 0.0
        return self._penalty(env, contact, command_name, max_time)

    def _penalty(self, env: ManagerBasedEnv, contact: torch.Tensor, command_name: str, max_time: float) -> torch.Tensor:
        """Advance the phase timers with the current foot contacts and return the penalty [s²]."""
        changed = (contact != self._contact) & self._initialized
        complete = changed & self._known
        self._air.copy_(torch.where(complete & contact, self._elapsed, self._air))
        self._stance.copy_(torch.where(complete & ~contact, self._elapsed, self._stance))
        self._elapsed.copy_(torch.where(changed, 0.0, self._elapsed) + env.step_dt)
        self._known |= changed
        self._contact.copy_(contact)
        self._initialized.fill_(True)
        penalty = torch.zeros(env.num_envs, device=env.device)
        for duration in (self._air, self._stance):
            penalty += duration.clamp(max=max_time).var(dim=1, correction=0) * (duration > 0.0).all(dim=1)
        return penalty * (env.command_manager.get_command(command_name)[:, :2].norm(dim=1) > 0.1)


class GeometryFeetAirTimeVariance(feet_air_time_variance):
    """:class:`feet_air_time_variance` with foot contacts from NeRD Newton-native contact geometry."""

    _feet_param = "body_cfg"

    def __call__(  # type: ignore[override]
        self,
        env: ManagerBasedEnv,
        command_name: str,
        body_cfg: SceneEntityCfg,
        max_time: float = 0.5,
        maximum_separation: float = 0.0,
    ) -> torch.Tensor:
        """Update the phase timers and return the penalty.

        Args:
            env: Environment with NeRD Newton-native contacts.
            command_name: Velocity command term name.
            body_cfg: Robot with the feet resolved.
            max_time: Duration clip [s].
            maximum_separation: Largest signed surface separation counted as contact [m].

        Returns:
            Penalty per environment [s²].
        """
        contact = body_contacts(body_cfg.body_ids, maximum_separation=maximum_separation, env=env)
        return self._penalty(env, contact, command_name, max_time)


def root_height_deficit_l2(env: ManagerBasedEnv, target_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the squared shortfall of the root height below ``target_height`` on flat ground [m²].

    Args:
        env: Environment.
        target_height: Root height above the environment origin [m]; heights above it are not penalized.
        asset_cfg: Robot.

    Returns:
        Penalty per environment.
    """
    height = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (target_height - height).clamp(min=0.0).square()


def root_height_below_minimum_after_warmup(
    env: ManagerBasedEnv, minimum_height: float, warmup_steps: int, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the root drops below ``minimum_height`` on flat ground, after ``warmup_steps`` env steps.

    Args:
        env: Environment.
        minimum_height: Root height above the environment origin [m].
        warmup_steps: Global environment steps before the term activates; resets do not restart it.
        asset_cfg: Robot.

    Returns:
        Termination flags per environment.
    """
    if env.common_step_counter < warmup_steps:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    height = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return height < minimum_height
