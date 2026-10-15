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
    prim_path="{ENV_REGEX_NS}/table",
    init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.55, 0.0, 0.005], rot=[0.0, 0.0, 0.0, 1.0]),
    spawn=sim_utils.CuboidCfg(
        size=(0.8, 1.5, 0.5),
        collision_props=CollisionPropertiesCfg(),
    ),
)
"""Static cuboid table used by Franka lift NeRD. It is not a Newton body.

The 0.8 m x 1.5 m top surface stays at z = 0.255 m, matching the core
``Isaac-Lift-Franka`` table and the official policy. The collider extends downward
to z = -0.245 m so cubes dropped by full-gravity reset cannot tunnel through the
thin 0.04 m core table. A kinematic rigid body would otherwise enter the NeRD
generalized state as a 13-DoF free joint.
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
    """Drop the Newton table articulation and disable Panda self-collisions.

    The robot spawner may carry one articulation-property config or a list of per-backend schema fragments; each
    fragment's own self-collision field is disabled.
    """
    env_cfg.scene.table = None
    props = env_cfg.scene.robot.spawn.articulation_props
    fragments = props if isinstance(props, list) else [props]
    updated = []
    for fragment in fragments:
        for field_name in ("enabled_self_collisions", "self_collision_enabled"):
            if hasattr(fragment, field_name):
                fragment = fragment.replace(**{field_name: False})
        updated.append(fragment)
    env_cfg.scene.robot.spawn.articulation_props = updated if isinstance(props, list) else updated[0]


def apply_franka_lift_nerd_scene(env_cfg) -> None:
    """Fix the core Lift object to the cube variant and replace the kinematic table with a static collider.

    The core task's ``ObjectCfg`` preset defaults to per-environment random shapes;
    NeRD state and contact-token layouts require one rigid object with fixed geometry.
    """
    from isaaclab_tasks.core.lift.lift_env_cfg import ObjectCfg

    env_cfg.scene.table = FRANKA_LIFT_TABLE_CFG.copy()
    env_cfg.scene.object.spawn = ObjectCfg().cube
    env_cfg.commands.object_pose.debug_vis = False
    clearance = env_cfg.events.conditional_reset.params["valid_criteria"]["robot_table_clearance"]
    clearance.body_names = ["panda_link[2-7]", "panda_hand", ".*finger"]


FRANKA_LIFT_RANDOMIZATION_EVENTS = (
    "robot_physics_material",
    "object_physics_material",
    "object_physics_inertia",
    "joint_stiffness_and_damping",
    "joint_friction",
    "object_scale_mass",
    "variable_gravity",
    "gripper_closing_speed",
)
"""Core Lift domain-randomization event terms that change the ground-truth dynamics."""


def apply_franka_lift_nominal_physics(env_cfg) -> None:
    """Remove domain randomization and the ADR curriculum so every world runs the nominal dynamics.

    The core Lift task scales drive gains, joint friction, contact materials, object
    mass/inertia and gravity per world and schedules gravity from zero with ADR. A
    NeRD model represents one dynamics, so recording and inference use the asset
    values and full gravity. Reset-bank sampling and observation noise stay as the
    policy expects.
    """
    for name in FRANKA_LIFT_RANDOMIZATION_EVENTS:
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    env_cfg.curriculum = None
    # Final ADR value: with full gravity the object may rest on the table (z = 0.28 m).
    env_cfg.terminations.object_out_of_bound.params["in_bound_range"]["z"] = (0.0, 2.0)


def apply_franka_lift_robot(env_cfg) -> None:
    """Install the core Lift Franka (Menagerie asset, Lift-calibrated drives) at ``{ENV_REGEX_NS}/Robot``."""
    from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import FRANKA_PANDA_LIFT_CFG

    env_cfg.scene.robot = FRANKA_PANDA_LIFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
