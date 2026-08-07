# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry-based MDP terms for NeRD Newton-native contacts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def _native_contact_adapter():
    """Return the active NeRD native-contact adapter."""
    from isaaclab_neural.physics import NewtonNerdManager

    adapter = NewtonNerdManager._nerd_contact_adapter
    if adapter is None or NewtonNerdManager._nerd_contact_mode != "newton_native":
        raise RuntimeError("Geometry contact MDP terms require NeRD contact_mode='newton_native'.")
    return adapter


def body_contacts(
    body_ids: Sequence[int] | slice,
    *,
    maximum_separation: float = 0.0,
) -> torch.Tensor:
    """Return geometric contact state for selected robot bodies.

    A native collision candidate counts as contact when its signed surface
    separation is at most ``maximum_separation``.

    Args:
        body_ids: Robot-local body indices.
        maximum_separation: Largest signed surface separation [m] considered contact.

    Returns:
        Contact flags shaped ``(num_envs, num_selected_bodies)``.
    """
    adapter = _native_contact_adapter()
    if isinstance(body_ids, slice):
        selected = torch.arange(adapter.bodies_per_env, device=adapter.device)[body_ids]
    else:
        selected = torch.as_tensor(body_ids, dtype=torch.long, device=adapter.device)

    active = adapter.contact_masks & (adapter.contact_depths <= maximum_separation)
    side0 = adapter.contact_body_ids_0.unsqueeze(-1) == selected
    side1 = adapter.contact_body_ids_1.unsqueeze(-1) == selected
    return torch.any(active.unsqueeze(-1) & (side0 | side1), dim=1)


class GeometryFeetAirTime(ManagerTermBase):
    """Reward foot air time at geometry-detected touchdown."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        body_cfg: SceneEntityCfg = cfg.params["body_cfg"]
        num_bodies = len(body_cfg.body_ids)
        self._air_time = torch.zeros((env.num_envs, num_bodies), device=env.device)
        self._previous_contact = torch.zeros((env.num_envs, num_bodies), dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._air_time[env_ids] = 0.0
        self._previous_contact[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        body_cfg: SceneEntityCfg,
        command_name: str,
        threshold: float,
        maximum_separation: float = 0.0,
    ) -> torch.Tensor:
        """Compute touchdown air-time reward from native contact geometry."""
        contact = body_contacts(body_cfg.body_ids, maximum_separation=maximum_separation)
        first_contact = contact & ~self._previous_contact & (self._air_time > 0.0)
        reward = torch.sum((self._air_time - threshold) * first_contact, dim=1)
        reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

        self._air_time = torch.where(contact, 0.0, self._air_time + env.step_dt)
        self._previous_contact.copy_(contact)
        return reward


def geometry_undesired_contacts(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    maximum_separation: float = 0.0,
) -> torch.Tensor:
    """Count selected bodies touching geometry."""
    del env
    return body_contacts(body_cfg.body_ids, maximum_separation=maximum_separation).sum(dim=1)


def geometry_illegal_contact(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    maximum_separation: float = 0.0,
) -> torch.Tensor:
    """Terminate when any selected body touches geometry."""
    del env
    return torch.any(body_contacts(body_cfg.body_ids, maximum_separation=maximum_separation), dim=1)
