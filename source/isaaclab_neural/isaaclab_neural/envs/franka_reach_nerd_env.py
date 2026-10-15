# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-free Franka Reach environments for NeRD data generation and inference."""

from __future__ import annotations

from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reach.config.franka.franka_reach_env_cfg import FrankaReachEnvCfg
from isaaclab_tasks.core.reach.reach_env_cfg import EventCfg as ReachEventCfg
from isaaclab_tasks.core.reach.reach_env_cfg import ReachPhysicsCfg

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg

from .franka_scene import FRANKA_PANDA_CFG, apply_franka_lift_robot, apply_franka_reach_contact_free_scene
from .incremental_joint_position import IncrementalJointPositionActionCfg, implicit_pd_rest_saturation_increment
from .neural_env_wrapper import NerdManagerBasedRLEnv
from .recording import apply_frame_stepping

DEFAULT_FRANKA_REACH_NERD_MODEL_PATH: str | None = None

_FRANKA_REACH_ARM_JOINTS = ["panda_joint.*"]
# Clamp the incremental walk to 60% of each USD joint range, leaving margin for
# transient overshoot under held saturation-scale targets.
FRANKA_REACH_DATASET_SOFT_JOINT_POS_LIMIT_FACTOR = 0.6


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
"""Historical increments [rad], retained for callers of the original data recipe.

Current data generation derives runtime drive values via ``scale=None`` instead.
"""
# Physics frames to hold each absolute ``q_target``. Reach uses ``dt=1/60`` and
# ``decimation=2``; 20 frames is 1/3 s.
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
        scale=None,
    )


FRANKA_REACH_PHYSICS_DT = 1.0 / 60.0
"""Physics step [s] of the Reach NeRD recipes and the Menagerie Reach teacher."""

FRANKA_REACH_DECIMATION = 2
"""Physics frames per 30 Hz Reach control step."""


def apply_franka_reach_timing(env_cfg) -> None:
    """Pin the 60 Hz physics and 30 Hz control timing the Reach teacher and datasets were recorded with.

    The core Reach task later moved to 120 Hz physics with decimation 4 and two MJWarp substeps; the Reach NeRD recipes
    keep their one-substep contract.
    """
    env_cfg.sim.dt = FRANKA_REACH_PHYSICS_DT
    env_cfg.decimation = FRANKA_REACH_DECIMATION
    env_cfg.sim.render_interval = FRANKA_REACH_DECIMATION
    newton_cfg = getattr(env_cfg.sim.physics, "newton_mjwarp", None)
    if newton_cfg is not None:
        newton_cfg.num_substeps = 1


@configclass
class FrankaReachDatasetGenEnvCfg(FrankaReachEnvCfg):
    """Franka Reach config for contact-free NeRD dataset generation.

    This keeps the Newton ``newton_mjwarp`` physics preset from
    :class:`~isaaclab_tasks.core.reach.reach_env_cfg.ReachEnvCfg`,
    removes the table articulation, and disables Panda self-collisions so the
    scene has one articulated robot per world.

    Random splits command ``q_target = clip(q + scale a, soft limits)`` with a
    per-joint ``scale`` of ``effort_limit / kp`` from the resolved robot
    drives and hold that absolute target for 20 physics frames. The Menagerie
    Panda runs at the Reach task's 60 Hz physics step. At rest, ``|a| = 1``
    therefore reaches each actuator's effort limit.
    Random splits reset arm positions uniformly within the soft joint limits,
    with zero joint velocities and fingers at their clamped default positions.
    Policy and zero-action splits restore the Reach action and reset through
    :meth:`restore_reach_rl_arm_action`. Policy clips resample the EE pose every
    :data:`FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S` seconds.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_franka_lift_robot(self)
        apply_franka_reach_contact_free_scene(self)
        apply_franka_reach_timing(self)
        self.scene.robot.soft_joint_pos_limit_factor = FRANKA_REACH_DATASET_SOFT_JOINT_POS_LIMIT_FACTOR
        self.actions.arm_action = franka_reach_dataset_arm_action()
        finger_pos = self.scene.robot.init_state.joint_pos["panda_finger_joint.*"]
        self.events.reset_robot_joints = EventTermCfg(
            func=mdp.reset_joints_within_limits_range,
            mode="reset",
            params={
                "position_range": {
                    "panda_joint.*": (None, None),
                    "panda_finger_joint.*": (finger_pos, finger_pos),
                },
                "velocity_range": {".*": (0.0, 0.0)},
            },
        )
        resample_s = FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S
        self.commands.ee_pose.resampling_time_range = (resample_s, resample_s)
        # Headless dataset generation does not need command markers.
        self.commands.ee_pose.debug_vis = False

    def restore_reach_rl_arm_action(self) -> None:
        """Restore the Reach RL action and reset used by policy and zero-action splits."""
        self.actions.arm_action = franka_reach_rl_arm_action()
        self.events.reset_robot_joints = ReachEventCfg().reset_robot_joints


@configclass
class FrankaReachPolicyEnvCfg(FrankaReachDatasetGenEnvCfg):
    """Dataset-Gen scene with the Reach RL action and reset, for training the control policy.

    This is the environment the policy and zero-action dataset splits run in, so a
    policy trained here matches the recorded action contract exactly.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.restore_reach_rl_arm_action()
        # Ordinary Reach goal timing; Dataset-Gen shortens it for policy clips only.
        self.commands.ee_pose.resampling_time_range = (4.0, 4.0)


@configclass
class FrankaReachControlStepDatasetGenEnvCfg(FrankaReachPolicyEnvCfg):
    """Reach policy environment stepped frame by frame for the env-driven control-step collector."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_frame_stepping(self)


@configclass
class NerdFrankaReachEnvCfg(FrankaReachEnvCfg):
    """Manager-based Franka Reach config using contact-free NeRD Newton physics."""

    # Upstream :class:`ReachPhysicsCfg` kept so Hydra ``presets=newton_mjwarp``
    # still resolves after ``sim.physics`` is replaced with :class:`NerdNewtonCfg`.
    physics_presets: ReachPhysicsCfg | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_franka_lift_robot(self)
        apply_franka_reach_contact_free_scene(self)
        apply_franka_reach_timing(self)
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
