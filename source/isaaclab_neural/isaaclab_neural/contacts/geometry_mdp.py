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

from .contact_set_schema import (
    CONTACT_REPRESENTATION_TOKENS,
    CONTACT_TOKEN_BODY_SLOT_INDEX,
    CONTACT_TOKEN_GAP_INDEX,
    CONTACT_TOKEN_VALID_INDEX,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def _native_contact_adapter():
    """Return the active NeRD native-contact adapter."""
    from isaaclab_neural.physics import NewtonNerdManager

    adapter = NewtonNerdManager._nerd_contact_adapter
    contact_mode = getattr(NewtonNerdManager, "_nerd_contact_mode", "newton_native")
    if adapter is None or contact_mode != "newton_native":
        raise RuntimeError("Geometry contact MDP terms require NeRD contact_mode='newton_native'.")
    return adapter


def _token_owner_local_ids(adapter, tokens: torch.Tensor) -> torch.Tensor:
    """Map packed tokens to Isaac-Lab local body indices.

    Encoder ``body_slot`` is a compact primary-body index and can disagree with
    :class:`~isaaclab.managers.SceneEntityCfg` ``body_ids``. Prefer packed Newton
    body ids converted through ``body_local_ids`` when the adapter provides them.
    """
    token_body_ids = getattr(adapter, "contact_token_body_ids", None)
    local_lookup = getattr(adapter, "body_local_ids", None)
    if token_body_ids is None or local_lookup is None:
        return tokens[..., CONTACT_TOKEN_BODY_SLOT_INDEX].to(dtype=torch.long)

    safe = token_body_ids.clamp(min=0)
    local_ids = local_lookup[safe]
    return torch.where(token_body_ids >= 0, local_ids, torch.full_like(local_ids, -1))


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

    if getattr(adapter, "contact_representation", None) == CONTACT_REPRESENTATION_TOKENS:
        tokens = adapter.contact_tokens
        active = (tokens[..., CONTACT_TOKEN_VALID_INDEX] > 0.5) & (
            tokens[..., CONTACT_TOKEN_GAP_INDEX] <= maximum_separation
        )
        owner_local_ids = _token_owner_local_ids(adapter, tokens)
        selected_bodies = owner_local_ids.unsqueeze(-1) == selected
        return torch.any(active.unsqueeze(-1) & selected_bodies, dim=1)

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
