# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anymal-C flat dataset-generation environment config."""

from __future__ import annotations

from copy import deepcopy

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg

from isaaclab_assets.robots.anymal import ANYDRIVE_3_SIMPLE_ACTUATOR_CFG


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        root_height = ObsTerm(
            func=base_mdp.base_pos_z,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


ROOT_HEIGHT_MINIMUM = 0.4
KP_RANGE = (30.0, 200.0)
KD_RANGE = (0.0, 4.0)


@configclass
class AnymalCDatasetGenFlatEnvCfg(AnymalCFlatEnvCfg):
    """Anymal-C flat config for NeRD dataset generation.

    This keeps the ground-truth Newton physics preset from
    :class:`AnymalCFlatEnvCfg`, but swaps the actuator to a simple explicit
    actuator so generated training data is not tied to the deployment LSTM
    actuator.
    """

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Disable contact-sensor related reward/termination terms (not working with NeRD solver)
        setattr(self.terminations, "base_contact", None)
        setattr(self.rewards, "feet_air_time", None)
        setattr(self.rewards, "undesired_contacts", None)
        # Disable random pushing event
        setattr(self.events, "push_robot", None)

        # Add root-height termination
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

        # Replacing the whole "legs" actuator cfg resets joint armature to the
        # DCMotor default; restore the preset-resolved armature (newton=0.01) set
        # by super() so data-gen physics matches the deployment env
        # (Anymal-C-Velocity-Flat), which NeRD is trained against.
        armature = self.scene.robot.actuators["legs"].armature
        self.scene.robot.actuators["legs"] = deepcopy(ANYDRIVE_3_SIMPLE_ACTUATOR_CFG)
        self.scene.robot.actuators["legs"].armature = armature


@configclass
class AnymalCDatasetGenRoughEnvCfg(AnymalCRoughEnvCfg):
    """Anymal-C rough config for NeRD dataset generation.

    This keeps the upstream rough-terrain observation and terrain curriculum
    setup so policy-driven collection can use rough-terrain policies, while
    replacing the actuator with a simple explicit actuator for data generation.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        # Disable contact-sensor related reward/termination terms (not working with NeRD solver)
        setattr(self.terminations, "base_contact", None)
        setattr(self.rewards, "feet_air_time", None)
        setattr(self.rewards, "undesired_contacts", None)
        # Disable random pushing event
        setattr(self.events, "push_robot", None)

        # Add root-height termination
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

        armature = self.scene.robot.actuators["legs"].armature
        self.scene.robot.actuators["legs"] = deepcopy(ANYDRIVE_3_SIMPLE_ACTUATOR_CFG)
        self.scene.robot.actuators["legs"].armature = armature
