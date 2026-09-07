# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 dataset-generation environment configs."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

from isaaclab_neural.robots.g1 import apply_g1_rich_velocity_commands

ROOT_HEIGHT_MINIMUM = 0.4
_JOINT_POSITION_OFFSET_RANGE = (-0.2, 0.2)
_JOINT_VELOCITY_OFFSET_RANGE = (-0.1, 0.1)
_BASE_HEIGHT_OFFSET_RANGE = (-0.1, 0.1)


def _configure_reset_disturbances(cfg) -> None:
    """Configure G1 reset disturbances used only for dataset generation."""
    cfg.events.reset_robot_joints.func = base_mdp.reset_joints_by_offset
    cfg.events.reset_robot_joints.params = {
        "position_range": _JOINT_POSITION_OFFSET_RANGE,
        "velocity_range": _JOINT_VELOCITY_OFFSET_RANGE,
    }
    cfg.events.reset_base.params["pose_range"]["z"] = _BASE_HEIGHT_OFFSET_RANGE


@configclass
class G1DatasetGenFlatEnvCfg(G1FlatEnvCfg):
    """G1 flat config for NeRD dataset generation.

    Keeps stock Newton physics and Implicit actuators. Disables contact-force
    sensor MDP terms that are unavailable or unreliable during data collection
    with the teacher solver presets used by NeRD pipelines.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_g1_rich_velocity_commands(self)
        _configure_reset_disturbances(self)
        setattr(self.terminations, "base_contact", None)
        setattr(self.rewards, "feet_air_time", None)
        setattr(self.rewards, "feet_slide", None)
        setattr(self.events, "push_robot", None)
        setattr(
            self.terminations,
            "low_root_height",
            DoneTerm(
                func=base_mdp.root_height_below_minimum,
                params={
                    "minimum_height": ROOT_HEIGHT_MINIMUM,
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
        )


@configclass
class G1DatasetGenRoughEnvCfg(G1RoughEnvCfg):
    """G1 rough config for NeRD dataset generation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_g1_rich_velocity_commands(self)
        _configure_reset_disturbances(self)
        setattr(self.terminations, "base_contact", None)
        setattr(self.rewards, "feet_air_time", None)
        setattr(self.rewards, "feet_slide", None)
        setattr(self.events, "push_robot", None)
        setattr(
            self.terminations,
            "low_root_height",
            DoneTerm(
                func=base_mdp.root_height_below_minimum,
                params={
                    "minimum_height": ROOT_HEIGHT_MINIMUM,
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
        )
