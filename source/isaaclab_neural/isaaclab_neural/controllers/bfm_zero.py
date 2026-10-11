# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BFM-Zero behavioural foundation model as a latent-conditioned low-level controller for the G1 29-DoF robot.

BFM-Zero (Li et al., ICLR 2026) maps a 256-dimensional latent ``z`` and proprioception to joint position targets
at 50 Hz. This module re-implements its residual actor in plain PyTorch and reproduces the observation convention
of the humanoidverse training environment, so that exported actor weights run inside Isaac Lab without the
BFM-Zero code base. The weights are licensed CC BY-NC 4.0 and are loaded from a user-provided file at runtime.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import string as string_utils
from isaaclab.utils.configclass import configclass

from isaaclab_neural.robots.g1_29dof import (
    G1_29DOF_DEFAULT_JOINT_POS,
    G1_29DOF_EFFORT_LIMIT,
    G1_29DOF_JOINT_NAMES,
    G1_29DOF_STIFFNESS,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

BFM_ZERO_ACTOR_PATH_ENV_VAR = "BFM_ZERO_ACTOR_PATH"
"""Environment variable naming the exported actor file used when the action config leaves the path unset."""

BFM_ZERO_Z_DIM = 256
"""Latent dimension."""

BFM_ZERO_Z_NORM = 16.0
"""Latent norm after projection, ``sqrt(BFM_ZERO_Z_DIM)``."""

BFM_ZERO_ACTION_RESCALE = 5.0
"""Multiplier from the actor's ``[-1, 1]`` output to the action stored in observations."""

BFM_ZERO_ACTION_SCALE = 0.25
"""Per-joint target offset is ``BFM_ZERO_ACTION_SCALE * effort_limit / stiffness`` times the rescaled action."""

BFM_ZERO_ANG_VEL_SCALE = 0.25
"""Scale applied to the base angular velocity observation."""

BFM_ZERO_HISTORY_LENGTH = 4
"""Number of previous control steps stored in the actor history."""

BFM_ZERO_HISTORY_KEYS = ("actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")
"""History terms, in the alphabetical order used by humanoidverse."""


def project_bfm_zero_latent(z: torch.Tensor) -> torch.Tensor:
    """Project latents onto the sphere of radius :data:`BFM_ZERO_Z_NORM`.

    Args:
        z: Latents, shape ``[..., BFM_ZERO_Z_DIM]``.

    Returns:
        Projected latents with the same shape.
    """
    return z * (BFM_ZERO_Z_NORM / z.norm(dim=-1, keepdim=True).clamp_min(1e-8))


class _Block(nn.Module):
    """LayerNorm, linear map and optional Mish activation."""

    def __init__(self, input_dim: int, output_dim: int, activation: bool):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim), *([nn.Mish()] if activation else [])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _ResidualBlock(nn.Module):
    """Pre-norm residual block with Mish activation."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Mish())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


def _embedding(input_dim: int, hidden_dim: int, layers: int) -> nn.Sequential:
    blocks = [_Block(input_dim, hidden_dim, True)] + [_ResidualBlock(hidden_dim) for _ in range(layers - 2)]
    return nn.Sequential(*blocks, _Block(hidden_dim, hidden_dim // 2, True))


class BfmZeroActor(nn.Module):
    """Deterministic BFM-Zero residual actor with a frozen observation normalizer.

    The layer layout and parameter names match the ``_actor`` module of the BFM-Zero ``FBcprAuxModel``, so its
    state dictionary loads without renaming. The batch-norm observation normalizer is folded into ``obs_mean`` and
    ``obs_var`` buffers.
    """

    def __init__(
        self,
        obs_dim: int = 465,
        z_dim: int = BFM_ZERO_Z_DIM,
        action_dim: int = 29,
        hidden_dim: int = 2048,
        hidden_layers: int = 6,
        embedding_layers: int = 2,
        norm_eps: float = 1e-5,
    ):
        """Initialize the actor.

        Args:
            obs_dim: Concatenated ``state``, ``last_action`` and ``history_actor`` dimension.
            z_dim: Latent dimension.
            action_dim: Number of actuated joints.
            hidden_dim: Width of the residual trunk.
            hidden_layers: Number of residual blocks in the policy head.
            embedding_layers: Number of blocks in each embedding; at least two.
            norm_eps: Variance epsilon of the observation normalizer.
        """
        super().__init__()
        if embedding_layers < 2:
            raise ValueError("BFM-Zero embeddings need at least two layers.")
        self.embed_z = _embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = _embedding(obs_dim, hidden_dim, embedding_layers)
        self.policy = nn.Sequential(
            *[_ResidualBlock(hidden_dim) for _ in range(hidden_layers)], _Block(hidden_dim, action_dim, False)
        )
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.norm_eps = norm_eps

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute the mean action.

        Args:
            obs: Unnormalized observations, shape ``[N, obs_dim]``.
            z: Projected latents, shape ``[N, z_dim]``.

        Returns:
            Actions in ``[-1, 1]``, shape ``[N, action_dim]``.
        """
        obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.norm_eps)
        embedding = torch.cat([self.embed_s(obs), self.embed_z(torch.cat([obs, z], dim=-1))], dim=-1)
        return torch.tanh(self.policy(embedding))


_ACTOR_META_KEYS = ("obs_dim", "z_dim", "action_dim", "hidden_dim", "hidden_layers", "embedding_layers", "norm_eps")


def load_bfm_zero_actor(path: str, device: str | torch.device) -> BfmZeroActor:
    """Load an exported BFM-Zero actor.

    The file is a ``torch.save`` dictionary with ``"actor"`` (state dictionary of :class:`BfmZeroActor`) and
    ``"meta"`` (constructor arguments ``obs_dim``, ``z_dim``, ``action_dim``, ``hidden_dim``, ``hidden_layers``,
    ``embedding_layers`` and ``norm_eps``).

    Args:
        path: Exported actor file.
        device: Device of the returned module.

    Returns:
        Frozen actor in evaluation mode.

    Raises:
        KeyError: The file lacks the actor state dictionary or metadata.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    if "actor" not in payload or "meta" not in payload:
        raise KeyError(f"{path} is not an exported BFM-Zero actor (missing 'actor' or 'meta').")
    meta = payload["meta"]
    actor = BfmZeroActor(**{key: meta[key] for key in _ACTOR_META_KEYS if key in meta})
    actor.load_state_dict(payload["actor"])
    return actor.to(device).eval().requires_grad_(False)


def bfm_zero_action_scale(joint_names: Sequence[str]) -> list[float]:
    """Return the per-joint target scale ``BFM_ZERO_ACTION_SCALE * effort_limit / stiffness``.

    Args:
        joint_names: Joint names in action order.

    Returns:
        Target offset per unit rescaled action [rad].
    """
    _, _, effort = string_utils.resolve_matching_names_values(G1_29DOF_EFFORT_LIMIT, joint_names, preserve_order=False)
    _, _, stiffness = string_utils.resolve_matching_names_values(G1_29DOF_STIFFNESS, joint_names, preserve_order=False)
    return [BFM_ZERO_ACTION_SCALE * e / k for e, k in zip(effort, stiffness)]


class BfmZeroObservationHistory:
    """Builds BFM-Zero actor observations with the humanoidverse training convention.

    The observation is ``[state, last_action, history_actor]`` with ``state = [q - q_default, qd, gravity_b,
    0.25 * omega_b]``. ``history_actor`` concatenates :data:`BFM_ZERO_HISTORY_KEYS`, each holding the previous
    :data:`BFM_ZERO_HISTORY_LENGTH` control steps newest first. The current step is appended to the history only
    after the observation is built, as in the training environment.
    """

    def __init__(self, num_envs: int, num_joints: int, device: str | torch.device):
        """Initialize empty histories.

        Args:
            num_envs: Number of environments.
            num_joints: Number of actuated joints.
            device: Tensor device.
        """
        dims = {
            "actions": num_joints,
            "base_ang_vel": 3,
            "dof_pos": num_joints,
            "dof_vel": num_joints,
            "projected_gravity": 3,
        }
        self.history = {
            key: torch.zeros(num_envs, BFM_ZERO_HISTORY_LENGTH, dims[key], device=device)
            for key in BFM_ZERO_HISTORY_KEYS
        }
        self.last_action = torch.zeros(num_envs, num_joints, device=device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear histories and the previous action.

        Args:
            env_ids: Environments to clear. Defaults to all.
        """
        env_ids = slice(None) if env_ids is None else env_ids
        for value in self.history.values():
            value[env_ids] = 0.0
        self.last_action[env_ids] = 0.0

    def observe(
        self,
        joint_pos_rel: torch.Tensor,
        joint_vel: torch.Tensor,
        projected_gravity: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
    ) -> torch.Tensor:
        """Build the actor observation and append the current step to the history.

        Args:
            joint_pos_rel: Joint positions minus defaults [rad], shape ``[N, J]``.
            joint_vel: Joint velocities [rad/s], shape ``[N, J]``.
            projected_gravity: Unit gravity direction in the base frame, shape ``[N, 3]``.
            root_ang_vel_b: Base angular velocity in the base frame [rad/s], shape ``[N, 3]``.

        Returns:
            Observations, shape ``[N, (2 * J + 6) + J + 4 * (3 * J + 6)]``, i.e. 465 for 29 joints.
        """
        ang_vel = root_ang_vel_b * BFM_ZERO_ANG_VEL_SCALE
        frame = {
            "actions": self.last_action,
            "base_ang_vel": ang_vel,
            "dof_pos": joint_pos_rel,
            "dof_vel": joint_vel,
            "projected_gravity": projected_gravity,
        }
        state = torch.cat([joint_pos_rel, joint_vel, projected_gravity, ang_vel], dim=-1)
        history = torch.cat([self.history[key].flatten(1) for key in BFM_ZERO_HISTORY_KEYS], dim=-1)
        obs = torch.cat([state, self.last_action, history], dim=-1)
        for key in BFM_ZERO_HISTORY_KEYS:
            self.history[key] = torch.cat([frame[key][:, None], self.history[key][:, :-1]], dim=1)
        return obs

    def record(self, action: torch.Tensor) -> None:
        """Store the rescaled action used as ``last_action`` at the next step.

        Args:
            action: Rescaled actions in ``[-BFM_ZERO_ACTION_RESCALE, BFM_ZERO_ACTION_RESCALE]``, shape ``[N, J]``.
        """
        self.last_action = action.clone()


class BfmZeroLatentAction(ActionTerm):
    """Action term whose input is a BFM-Zero latent; the frozen actor produces joint position targets.

    Latents are projected to norm :data:`BFM_ZERO_Z_NORM`. The actor runs every ``control_decimation`` physics
    steps, so an environment step may span several actor steps with one latent.
    """

    cfg: BfmZeroLatentActionCfg
    _asset: Articulation

    def __init__(self, cfg: BfmZeroLatentActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        joint_ids, joint_names = self._asset.find_joints(list(G1_29DOF_JOINT_NAMES), preserve_order=True)
        if list(joint_names) != list(G1_29DOF_JOINT_NAMES):
            raise ValueError(f"Articulation joints {joint_names} do not match the BFM-Zero joint order.")
        self._joint_ids = torch.as_tensor(joint_ids, device=self.device, dtype=torch.long)
        path = cfg.actor_path or os.environ.get(BFM_ZERO_ACTOR_PATH_ENV_VAR)
        if not path:
            raise ValueError(f"Set BfmZeroLatentActionCfg.actor_path or {BFM_ZERO_ACTOR_PATH_ENV_VAR}.")
        self._actor = load_bfm_zero_actor(path, self.device)
        default_ids, _, default_values = string_utils.resolve_matching_names_values(
            cfg.default_joint_pos, G1_29DOF_JOINT_NAMES, strict=False
        )
        self._default = torch.zeros(len(G1_29DOF_JOINT_NAMES), device=self.device)
        self._default[default_ids] = torch.tensor(default_values, device=self.device)
        self._scale = torch.tensor(bfm_zero_action_scale(G1_29DOF_JOINT_NAMES), device=self.device)
        self._history = BfmZeroObservationHistory(self.num_envs, len(G1_29DOF_JOINT_NAMES), self.device)
        self._raw_actions = torch.zeros(self.num_envs, BFM_ZERO_Z_DIM, device=self.device)
        self._latents = torch.zeros_like(self._raw_actions)
        self._targets = self._default.repeat(self.num_envs, 1)
        self._physics_step = 0

    @property
    def action_dim(self) -> int:
        return BFM_ZERO_Z_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._latents

    @property
    def joint_targets(self) -> torch.Tensor:
        """Latest joint position targets [rad], shape ``[N, 29]``, in BFM-Zero joint order."""
        return self._targets

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._latents = project_bfm_zero_latent(actions) if self.cfg.project_latent else actions.clone()

    @torch.no_grad()
    def apply_actions(self) -> None:
        if self._physics_step % self.cfg.control_decimation == 0:
            data = self._asset.data
            joint_pos = data.joint_pos.torch[:, self._joint_ids]
            obs = self._history.observe(
                joint_pos - self._default,
                data.joint_vel.torch[:, self._joint_ids],
                data.projected_gravity_b.torch,
                data.root_ang_vel_b.torch,
            )
            action = self._actor(obs, self._latents).clamp(-1.0, 1.0) * BFM_ZERO_ACTION_RESCALE
            if self.cfg.action_noise_std > 0.0:
                noise = self.cfg.action_noise_std * torch.randn_like(action)
                action = (action + noise).clamp(-BFM_ZERO_ACTION_RESCALE, BFM_ZERO_ACTION_RESCALE)
            self._history.record(action)
            limits = data.joint_pos_limits.torch[:, self._joint_ids]
            self._targets = torch.clamp(self._default + self._scale * action, limits[..., 0], limits[..., 1])
        self._physics_step += 1
        self._asset.actuators.target_command.set_position_index(value=self._targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._history.reset(env_ids)
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0


@configclass
class BfmZeroLatentActionCfg(ActionTermCfg):
    """Configuration of :class:`BfmZeroLatentAction`."""

    class_type: type[ActionTerm] = BfmZeroLatentAction

    asset_name: str = "robot"

    actor_path: str | None = None
    """Exported actor file; falls back to :data:`BFM_ZERO_ACTOR_PATH_ENV_VAR` when None."""

    control_decimation: int = 4
    """Physics steps per actor step; 4 gives the 50 Hz BFM-Zero control rate at 200 Hz physics."""

    project_latent: bool = True
    """Whether to project input latents to norm :data:`BFM_ZERO_Z_NORM`."""

    action_noise_std: float = 0.0
    """Standard deviation of Gaussian noise added to the rescaled action at each actor step.

    The noisy action is applied and fed back as ``last_action``, as for a perturbed command on hardware.
    """

    default_joint_pos: dict[str, float] = G1_29DOF_DEFAULT_JOINT_POS
    """Default joint positions of the BFM-Zero action and observation offsets [rad]."""
