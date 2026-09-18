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


class GeometryFeetAirTimePositiveBiped(ManagerTermBase):
    """Reward single-foot support with physics-step geometry contact timers.

    Timers sample the solver's input contacts after each physics step, so the
    contact state precedes the returned pose by one physics step. Reading the
    reward does not advance time. The physics manager clears the callback on close.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._body_ids = cfg.params["body_cfg"].body_ids
        self._maximum_separation = cfg.params.get("maximum_separation", 0.0)
        self._dt = env.physics_dt
        self._air_time = torch.zeros((env.num_envs, len(self._body_ids)), device=env.device)
        self._contact_time = torch.zeros_like(self._air_time)
        env.sim.physics_manager.register_post_step_callback(self._update)

    def _update(self) -> None:
        contact = body_contacts(self._body_ids, maximum_separation=self._maximum_separation)
        self._air_time.copy_(torch.where(contact, 0.0, self._air_time + self._dt))
        self._contact_time.copy_(torch.where(contact, self._contact_time + self._dt, 0.0))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._air_time[env_ids] = 0.0
        self._contact_time[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        body_cfg: SceneEntityCfg,
        command_name: str,
        threshold: float,
        maximum_separation: float = 0.0,
    ) -> torch.Tensor:
        """Compute the stock biped reward using geometric contact durations.

        Args:
            env: Environment supplying velocity commands.
            body_cfg: Feet selected when constructing the term.
            command_name: Velocity command to gate standing environments.
            threshold: Maximum rewarded single-support duration [s].
            maximum_separation: Contact separation configured at construction [m].

        Returns:
            Capped single-support duration [s] for each environment.
        """
        del body_cfg, maximum_separation
        contact = self._contact_time > 0.0
        mode_time = torch.where(contact, self._contact_time, self._air_time)
        single_stance = contact.sum(dim=1) == 1
        reward = torch.where(single_stance, mode_time.min(dim=1).values, 0.0).clamp(max=threshold)
        return reward * (env.command_manager.get_command(command_name)[:, :2].norm(dim=1) > 0.1)


class GeometryFeetSlide(ManagerTermBase):
    """Measure foot COM horizontal speed during recent geometry contact.

    Contact history samples the solver's input contacts at physics-step frequency.
    This is a force-free sliding proxy, not a measurement of contact-point slip.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._body_ids = cfg.params["body_cfg"].body_ids
        self._maximum_separation = cfg.params.get("maximum_separation", 0.0)
        history_length = cfg.params.get("history_length", 3)
        if history_length < 1:
            raise ValueError("GeometryFeetSlide requires history_length >= 1.")
        self._history = torch.zeros(
            (env.num_envs, history_length, len(self._body_ids)), dtype=torch.bool, device=env.device
        )
        env.sim.physics_manager.register_post_step_callback(self._update)

    def _update(self) -> None:
        self._history[:, 1:] = self._history[:, :-1].clone()
        self._history[:, 0] = body_contacts(self._body_ids, maximum_separation=self._maximum_separation)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._history[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        body_cfg: SceneEntityCfg,
        history_length: int = 3,
        maximum_separation: float = 0.0,
    ) -> torch.Tensor:
        """Sum planar foot speed over bodies with recent contact.

        Args:
            env: Environment containing the selected robot.
            body_cfg: Feet selected when constructing the term.
            history_length: Number of physics-step samples configured at construction.
            maximum_separation: Contact separation configured at construction [m].

        Returns:
            Sum of contacting foot COM horizontal speeds [m/s] per environment.
        """
        del history_length, maximum_separation
        velocity = env.scene[body_cfg.name].data.body_lin_vel_w.torch[:, body_cfg.body_ids, :2]
        return (velocity.norm(dim=-1) * self._history.any(dim=1)).sum(dim=1)


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
