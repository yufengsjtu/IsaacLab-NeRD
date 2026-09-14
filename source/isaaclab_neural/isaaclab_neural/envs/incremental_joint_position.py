# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Relative joint-position steps stored as held absolute PD targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import string as string_utils
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor


def implicit_pd_rest_saturation_increment(effort_limit: float, stiffness: float) -> float:
    """Return the joint step [rad] that saturates implicit PD at rest.

    At ``qd = 0``, ``tau = stiffness * Δq``. The step that reaches
    ``effort_limit`` is ``effort_limit / stiffness``.

    Args:
        effort_limit: Actuator effort limit [N or N·m].
        stiffness: Implicit PD stiffness [N/m or N·m/rad].

    Returns:
        Absolute joint increment [m or rad] at which rest torque equals the limit.
    """
    if stiffness <= 0.0:
        raise ValueError(f"stiffness must be positive, got {stiffness}.")
    return float(effort_limit) / float(stiffness)


def compute_incremental_position_targets(
    joint_pos: torch.Tensor,
    actions: torch.Tensor,
    scale: float | torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Map a relative action sample to an absolute joint-position target.

    Args:
        joint_pos: Current joint positions [rad], shape ``(N, J)``.
        actions: Raw actions in ``[-1, 1]``, shape ``(N, J)``.
        scale: Maximum joint step [rad]. Scalar or shape ``(N, J)``.
        lower: Soft lower limits [rad], shape ``(N, J)``.
        upper: Soft upper limits [rad], shape ``(N, J)``.

    Returns:
        Absolute position targets [rad], shape ``(N, J)``.
    """
    return torch.clamp(joint_pos + actions * scale, min=lower, max=upper)


class IncrementalJointPositionAction(ActionTerm):
    """Hold ``q_target = clip(q + scale * a, soft limits)`` until the next process.

    Unlike :class:`~isaaclab.envs.mdp.actions.joint_actions.RelativeJointPositionAction`,
    this term freezes the absolute target computed at process time. Holding the
    same raw action therefore lets PD track a fixed pose instead of chasing a
    receding offset. ``scale`` may be per-joint so each DoF can approach its
    own effort limit at ``|a| = 1``.
    """

    cfg: IncrementalJointPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: IncrementalJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=cfg.preserve_order)
        self._num_joints = len(self._joint_ids)
        if self._num_joints == self._asset.num_joints and not cfg.preserve_order:
            self._joint_ids = slice(None)
        self._scale = self._parse_scale(cfg.scale)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    def _parse_scale(self, scale: float | dict[str, float] | None) -> float | torch.Tensor:
        if scale is None:
            stiffness = self._asset.data.joint_stiffness.torch[:, self._joint_ids]
            effort_limit = self._asset.data.joint_effort_limits.torch[:, self._joint_ids]
            if not bool(torch.all(torch.isfinite(stiffness) & (stiffness > 0))):
                raise ValueError("Rest-saturation scaling requires finite positive joint stiffness.")
            if not bool(torch.all(torch.isfinite(effort_limit) & (effort_limit >= 0))):
                raise ValueError("Rest-saturation scaling requires finite non-negative joint effort limits.")
            return effort_limit / stiffness
        if isinstance(scale, (float, int)):
            return float(scale)
        if not isinstance(scale, dict):
            raise ValueError(f"Unsupported scale type: {type(scale)}. Supported values are float, dict, or None.")
        parsed = torch.full((self.num_envs, self.action_dim), float("nan"), device=self.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            scale, self._joint_names, preserve_order=self.cfg.preserve_order
        )
        parsed[:, index_list] = torch.tensor(value_list, device=self.device)
        uncovered = torch.isnan(parsed[0])
        if bool(uncovered.any()):
            missing = [
                name for name, is_missing in zip(self._joint_names, uncovered.tolist(), strict=True) if is_missing
            ]
            raise ValueError(f"scale dict did not cover all joints: {missing}.")
        return parsed

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "JointAction"
        self._IO_descriptor.joint_names = self._joint_names
        if isinstance(self._scale, torch.Tensor):
            self._IO_descriptor.scale = self._scale[0].detach().cpu().numpy().tolist()
        else:
            self._IO_descriptor.scale = self._scale
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        self._processed_actions[:] = compute_incremental_position_targets(
            joint_pos,
            self._raw_actions,
            self._scale,
            limits[..., 0],
            limits[..., 1],
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._asset.data.joint_pos.torch[env_ids][:, self._joint_ids]


@configclass
class IncrementalJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`IncrementalJointPositionAction`."""

    class_type: type[IncrementalJointPositionAction] = IncrementalJointPositionAction

    joint_names: list[str] = cast(list[str], MISSING)
    """Joint names or regex expressions that the action maps to."""

    scale: float | dict[str, float] | None = 0.5
    """Maximum joint step applied to the current position [rad].

    A dict of regex expressions sets a per-joint step, so ``|a| = 1`` can
    approach each actuator's effort limit independently. ``None`` derives
    per-environment, per-joint steps from the resolved articulation's effort
    limits divided by stiffness at action-term initialization, including
    USD-authored drives. This requires finite positive stiffness and finite
    non-negative effort limits for every controlled joint.
    """

    preserve_order: bool = False
    """Whether to preserve the configured joint-name order."""
