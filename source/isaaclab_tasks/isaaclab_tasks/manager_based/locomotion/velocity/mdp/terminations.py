# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _data_tensor(field) -> torch.Tensor | None:
    """Return a torch tensor from a plain tensor or ProxyArray-like field."""
    if field is None:
        return None
    return field.torch if hasattr(field, "torch") else field


def non_finite_articulation_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate environments whose articulation state contains NaN or Inf values."""
    asset: Articulation = env.scene[asset_cfg.name]

    invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for field_name in (
        "root_link_pose_w",
        "root_link_vel_w",
        "root_com_pose_w",
        "root_com_vel_w",
        "root_lin_vel_b",
        "root_ang_vel_b",
        "projected_gravity_b",
        "joint_pos",
        "joint_vel",
    ):
        field = getattr(asset.data, field_name, None)
        if field is None:
            continue
        tensor = field.torch if hasattr(field, "torch") else field
        if tensor is None or tensor.shape[0] != env.num_envs:
            continue
        invalid |= ~torch.isfinite(tensor.reshape(env.num_envs, -1)).all(dim=1)
    return invalid


def unstable_articulation_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_lin_vel_limit: float = 50.0,
    root_ang_vel_limit: float = 100.0,
    joint_vel_limit: float = 150.0,
) -> torch.Tensor:
    """Terminate environments whose articulation state is finite but physically unstable."""
    asset: Articulation = env.scene[asset_cfg.name]
    unstable = non_finite_articulation_state(env, asset_cfg)

    root_lin_vel = _data_tensor(getattr(asset.data, "root_lin_vel_b", None))
    if root_lin_vel is not None and root_lin_vel.shape[0] == env.num_envs:
        root_lin_vel_norm = torch.linalg.norm(root_lin_vel, dim=-1)
        unstable |= torch.isfinite(root_lin_vel_norm) & (root_lin_vel_norm > root_lin_vel_limit)

    root_ang_vel = _data_tensor(getattr(asset.data, "root_ang_vel_b", None))
    if root_ang_vel is not None and root_ang_vel.shape[0] == env.num_envs:
        root_ang_vel_norm = torch.linalg.norm(root_ang_vel, dim=-1)
        unstable |= torch.isfinite(root_ang_vel_norm) & (root_ang_vel_norm > root_ang_vel_limit)

    joint_vel = _data_tensor(getattr(asset.data, "joint_vel", None))
    if joint_vel is not None and joint_vel.shape[0] == env.num_envs:
        joint_vel_abs_max = torch.amax(torch.abs(joint_vel), dim=1)
        unstable |= torch.isfinite(joint_vel_abs_max) & (joint_vel_abs_max > joint_vel_limit)

    return unstable


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w.torch[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w.torch[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
