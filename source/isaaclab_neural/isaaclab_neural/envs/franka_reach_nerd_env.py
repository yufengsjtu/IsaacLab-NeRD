# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-free Franka Reach environments for NeRD data generation and inference."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reach.config.franka.franka_reach_env_cfg import FrankaReachEnvCfg
from isaaclab_tasks.core.reach.reach_env_cfg import ReachPhysicsCfg

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg

from .franka_scene import FRANKA_PANDA_CFG, apply_franka_reach_contact_free_scene
from .incremental_joint_position import IncrementalJointPositionActionCfg, implicit_pd_rest_saturation_increment
from .neural_env_wrapper import NerdManagerBasedRLEnv

DEFAULT_FRANKA_REACH_NERD_MODEL_PATH: str | None = None

_FRANKA_REACH_ARM_JOINTS = ["panda_joint.*"]
# Clamp the incremental walk to 80% of each USD joint range so underdamped PD
# overshoot still has margin before the hard limit.
FRANKA_REACH_DATASET_SOFT_JOINT_POS_LIMIT_FACTOR = 0.8


def _arm_saturation_increment_scale() -> dict[str, float]:
    """Return per-group ``|Δq|`` [rad] that saturates implicit PD at rest."""
    scales: dict[str, float] = {}
    for group_name in ("panda_shoulder", "panda_forearm"):
        actuator = FRANKA_PANDA_CFG.actuators[group_name]
        stiffness = actuator.stiffness
        effort = actuator.joint_effort_limit
        if not isinstance(stiffness, (int, float)) or not isinstance(effort, (int, float)):
            raise TypeError(
                f"{group_name} stiffness and joint_effort_limit must be scalars, got {stiffness!r} and {effort!r}."
            )
        increment = implicit_pd_rest_saturation_increment(float(effort), float(stiffness))
        for pattern in actuator.joint_names_expr:
            scales[pattern] = increment
    return scales


# Per-joint ``|Δq|`` [rad] so ``kp * Δq`` equals ``joint_effort_limit`` at rest.
FRANKA_REACH_DATASET_INCREMENT_SCALE = _arm_saturation_increment_scale()
# Physics frames to hold each absolute ``q_target``. Reach uses ``dt=1/120`` and
# ``decimation=4``; 20 frames is 1/6 s.
FRANKA_REACH_DATASET_ACTION_HOLD_FRAMES = 20
# Dataset-Gen EE pose resample [s]. Lets the Reach policy finish the fast
# approach (~1 s) and record a short settle before the next goal.
# ``Isaac-Reach-Franka-NeRD-v0`` keeps the 4 s Reach command for closed-loop
# eval.
FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S = 1.5


def franka_reach_rl_arm_action() -> JointPositionActionCfg:
    """Return the Reach RL arm action: ``q_default + 0.5 * a`` for ``a in [-1, 1]``."""
    return JointPositionActionCfg(
        asset_name="robot",
        joint_names=_FRANKA_REACH_ARM_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )


def franka_reach_dataset_arm_action() -> IncrementalJointPositionActionCfg:
    """Return a held incremental arm action for random dataset generation."""
    return IncrementalJointPositionActionCfg(
        asset_name="robot",
        joint_names=_FRANKA_REACH_ARM_JOINTS,
        scale=FRANKA_REACH_DATASET_INCREMENT_SCALE,
    )


@configclass
class FrankaReachDatasetGenEnvCfg(FrankaReachEnvCfg):
    """Franka Reach config for contact-free NeRD dataset generation.

    This keeps the Newton ``newton_mjwarp`` physics preset from
    :class:`~isaaclab_tasks.core.reach.reach_env_cfg.ReachEnvCfg`,
    removes the table articulation, and disables Panda self-collisions so the
    scene has one articulated robot per world.

    Random splits command ``q_target = clip(q + scale a, soft limits)`` with a
    per-joint ``scale`` of ``effort_limit / kp`` (shoulder ``1.0875`` rad,
    forearm ``0.15`` rad) and hold that absolute target for 20 physics frames.
    At rest, ``|a| = 1`` therefore reaches each actuator's effort limit.
    Policy and zero-action splits restore :meth:`restore_reach_rl_arm_action`.
    Policy clips resample the EE pose every
    :data:`FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S` seconds.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_franka_reach_contact_free_scene(self)
        self.scene.robot.soft_joint_pos_limit_factor = FRANKA_REACH_DATASET_SOFT_JOINT_POS_LIMIT_FACTOR
        self.actions.arm_action = franka_reach_dataset_arm_action()
        resample_s = FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S
        self.commands.ee_pose.resampling_time_range = (resample_s, resample_s)
        # Headless dataset generation does not need command markers.
        self.commands.ee_pose.debug_vis = False

    def restore_reach_rl_arm_action(self) -> None:
        """Restore the Reach RL joint-position action used by the trained policy."""
        self.actions.arm_action = franka_reach_rl_arm_action()


@configclass
class NerdFrankaReachEnvCfg(FrankaReachEnvCfg):
    """Manager-based Franka Reach config using contact-free NeRD Newton physics."""

    # Upstream :class:`ReachPhysicsCfg` kept so Hydra ``presets=newton_mjwarp``
    # still resolves after ``sim.physics`` is replaced with :class:`NerdNewtonCfg`.
    physics_presets: ReachPhysicsCfg | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_franka_reach_contact_free_scene(self)
        self.actions.arm_action = franka_reach_rl_arm_action()
        nerd_physics = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="TransformerNeuralSolver",
                neural_model_path=DEFAULT_FRANKA_REACH_NERD_MODEL_PATH,
                num_states_history=10,
                num_contacts_per_env=0,
                contact_mode="newton_native",
                contact_packing_policy="penetration_priority",
                states_frame="world",
                anchor_frame_step="every",
                states_embedding_type="identical",
                prediction_type="relative",
                orientation_prediction_parameterization="quaternion",
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )
        if isinstance(self.sim.physics, ReachPhysicsCfg):
            self.sim.physics.newton_mjwarp = nerd_physics
            self.physics_presets = self.sim.physics
        self.sim.physics = nerd_physics


class NerdFrankaReachEnv(NerdManagerBasedRLEnv):
    """Franka Reach env that installs ``NerdNewtonCfg`` before simulation init."""

    robot_name = "Franka"
