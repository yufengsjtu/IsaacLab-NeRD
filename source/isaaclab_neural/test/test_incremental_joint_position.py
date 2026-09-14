# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for incremental joint-position dataset-generation actions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from isaaclab_neural.envs.franka_reach_nerd_env import FRANKA_REACH_DATASET_INCREMENT_SCALE
from isaaclab_neural.envs.franka_scene import FRANKA_PANDA_CFG
from isaaclab_neural.envs.incremental_joint_position import (
    IncrementalJointPositionAction,
    IncrementalJointPositionActionCfg,
    compute_incremental_position_targets,
    implicit_pd_rest_saturation_increment,
)


def test_incremental_targets_add_scaled_action() -> None:
    joint_pos = torch.tensor([[0.0, 1.0]])
    actions = torch.tensor([[1.0, -0.5]])
    lower = torch.tensor([[-2.0, -2.0]])
    upper = torch.tensor([[2.0, 2.0]])

    targets = compute_incremental_position_targets(joint_pos, actions, scale=0.5, lower=lower, upper=upper)

    torch.testing.assert_close(targets, torch.tensor([[0.5, 0.75]]))


def test_incremental_targets_use_per_joint_scale() -> None:
    joint_pos = torch.zeros(1, 2)
    actions = torch.ones(1, 2)
    scale = torch.tensor([[1.0875, 0.15]])
    lower = torch.full((1, 2), -4.0)
    upper = torch.full((1, 2), 4.0)

    targets = compute_incremental_position_targets(joint_pos, actions, scale=scale, lower=lower, upper=upper)

    torch.testing.assert_close(targets, scale)


def test_incremental_targets_clamp_to_soft_limits() -> None:
    joint_pos = torch.tensor([[1.8]])
    actions = torch.tensor([[1.0]])
    lower = torch.tensor([[-2.0]])
    upper = torch.tensor([[2.0]])

    targets = compute_incremental_position_targets(joint_pos, actions, scale=0.5, lower=lower, upper=upper)

    torch.testing.assert_close(targets, torch.tensor([[2.0]]))


def test_rest_saturation_increment_is_effort_over_stiffness() -> None:
    assert implicit_pd_rest_saturation_increment(87.0, 80.0) == 87.0 / 80.0
    assert implicit_pd_rest_saturation_increment(12.0, 80.0) == 12.0 / 80.0


@pytest.mark.parametrize("stiffness", [40.0, 80.0, 0.0, -1.0, float("nan"), float("inf")])
def test_increment_scale_uses_resolved_drives(stiffness: float) -> None:
    position = torch.tensor([[0.0, 0.2, -0.1], [0.0, -0.2, 0.1]])
    gains = torch.tensor([[0.0, stiffness, 120.0], [0.0, stiffness * 2.0, 60.0]])
    efforts = torch.tensor([[0.0, 12.0, 24.0], [0.0, 24.0, 6.0]])
    asset = SimpleNamespace(
        num_joints=3,
        find_joints=lambda names, preserve_order: ([2, 1], ["joint2", "joint1"]),
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=position),
            joint_stiffness=SimpleNamespace(torch=gains),
            joint_effort_limits=SimpleNamespace(torch=efforts),
            soft_joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-4.0, 4.0]] * 3] * 2)),
        ),
    )
    env = SimpleNamespace(scene={"robot": asset}, num_envs=2, device="cpu")
    cfg = IncrementalJointPositionActionCfg(asset_name="robot", joint_names=["joint2", "joint1"], scale=None)
    if not 0.0 < stiffness < float("inf"):
        with pytest.raises(ValueError, match="stiffness"):
            IncrementalJointPositionAction(cfg, env)
        return
    action = IncrementalJointPositionAction(cfg, env)
    action.process_actions(torch.ones(2, 2))
    expected = position[:, [2, 1]] + efforts[:, [2, 1]] / gains[:, [2, 1]]
    torch.testing.assert_close(action.processed_actions, expected)


def test_franka_increment_scale_matches_actuator_saturation() -> None:
    shoulder = FRANKA_PANDA_CFG.actuators["panda_shoulder"]
    forearm = FRANKA_PANDA_CFG.actuators["panda_forearm"]
    assert isinstance(shoulder.stiffness, (int, float))
    assert isinstance(shoulder.joint_effort_limit, (int, float))
    assert isinstance(forearm.stiffness, (int, float))
    assert isinstance(forearm.joint_effort_limit, (int, float))
    assert FRANKA_REACH_DATASET_INCREMENT_SCALE["panda_joint[1-4]"] == implicit_pd_rest_saturation_increment(
        float(shoulder.joint_effort_limit), float(shoulder.stiffness)
    )
    assert FRANKA_REACH_DATASET_INCREMENT_SCALE["panda_joint[5-7]"] == implicit_pd_rest_saturation_increment(
        float(forearm.joint_effort_limit), float(forearm.stiffness)
    )
