# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Franka scene helpers for NeRD dataset generation and inference."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import CollisionPropertiesCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Panda spawn is inlined so OSMO (which installs only ``isaaclab_neural`` on the
# Isaac Lab image) does not inherit the image's ``isaaclab_assets`` USD path.
FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            joint_effort_limit=87.0,
            stiffness=80.0,
            damping=4.0,
            armature=1e-3,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            joint_effort_limit=12.0,
            stiffness=80.0,
            damping=4.0,
            armature=1e-3,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            joint_effort_limit=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Fixed-base Panda with the Isaac Lab Nucleus Legacy USD."""

FRANKA_LIFT_TABLE_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, -0.028], rot=[0.0, 0.0, 0.0, 1.0]),
    spawn=sim_utils.CuboidCfg(
        size=(0.9, 1.3, 0.05),
        collision_props=CollisionPropertiesCfg(),
    ),
)
"""Static cuboid table used by Franka lift NeRD. It is not an articulation.

The top surface is at z = -0.003 [m], matching the standard SeattleLabTable
collider so a resting cube stays below the Lift reward's height threshold.
"""


def panda_robot_cfg(*, enable_self_collisions: bool) -> ArticulationCfg:
    """Return the inlined Panda USD with an explicit self-collision flag.

    Args:
        enable_self_collisions: Whether Panda links may collide with each other.

    Returns:
        An :class:`~isaaclab.assets.ArticulationCfg` spawned at ``{ENV_REGEX_NS}/Robot``.
    """
    cfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    spawn = cfg.spawn.replace(
        articulation_props=cfg.spawn.articulation_props.replace(enabled_self_collisions=enable_self_collisions)
    )
    return cfg.replace(spawn=spawn)


def apply_franka_reach_contact_free_scene(env_cfg) -> None:
    """Drop the Newton table articulation and disable Panda self-collisions."""
    env_cfg.scene.table = None
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False


def apply_franka_lift_nerd_scene(env_cfg) -> None:
    """Keep the cube dynamic and replace the USD table with a static collider."""
    env_cfg.scene.table = FRANKA_LIFT_TABLE_CFG.copy()
