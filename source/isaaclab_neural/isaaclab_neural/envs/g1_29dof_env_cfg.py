# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 29-DoF environments driven by the BFM-Zero controller on Newton MJWarp."""

from __future__ import annotations

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_neural.controllers.bfm_zero import BfmZeroLatentActionCfg
from isaaclab_neural.envs.g1_dataset_gen_cfg import G1DatasetGenFlatEnvCfg
from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg, g1_invalid_state
from isaaclab_neural.envs.mdp_g1_29dof import (
    BaseHeightTorsoCommandCfg,
    apply_root_wrench_pulses,
    reset_root_lying,
    track_base_height_exp,
    track_torso_roll_pitch_exp,
)
from isaaclab_neural.robots.g1_29dof import G1_29DOF_BFM_CFG, G1_29DOF_WRIST_YAW_LINK_MASS

G1_29DOF_MJWARP_PHYSICS = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(
        njmax=600, nconmax=150, cone="pyramidal", impratio=1.0, integrator="implicitfast", use_mujoco_contacts=False
    ),
    collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
    num_substeps=2,
    debug_mode=False,
    default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
)
"""MJWarp settings of the stock G1 velocity task with constraint capacity for whole-body ground contact."""


@configclass
class G129DofPhysicsCfg(PresetCfg):
    """Newton MJWarp only; the ``newton_mjwarp`` name matches the preset selected by the NeRD dataset CLI."""

    newton_mjwarp = G1_29DOF_MJWARP_PHYSICS
    default = newton_mjwarp


@configclass
class G129DofSceneCfg(InteractiveSceneCfg):
    """Flat ground and one G1 29-DoF robot per environment."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    robot: ArticulationCfg = G1_29DOF_BFM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G129DofBfmActionsCfg:
    """BFM-Zero latent action."""

    bfm = BfmZeroLatentActionCfg(asset_name="robot")


@configclass
class G129DofBfmObservationsCfg:
    """Proprioceptive observations for logging; BFM-Zero builds its own actor observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class G129DofBfmEventsCfg:
    """Wrist mass correction, reset distribution and optional root pushes."""

    wrist_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wrist_yaw_link"),
            "mass_distribution_params": (G1_29DOF_WRIST_YAW_LINK_MASS, G1_29DOF_WRIST_YAW_LINK_MASS),
            "operation": "abs",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"yaw": (-math.pi, math.pi)}, "velocity_range": {}},
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0)},
    )
    reset_lying = EventTerm(func=reset_root_lying, mode="reset", params={"fraction": 0.0})
    actuator_gains: EventTerm | None = None
    root_pushes: EventTerm | None = None


@configclass
class G129DofBfmRewardsCfg:
    """Placeholder reward; the collection environment is not trained."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class G129DofBfmTerminationsCfg:
    """Timeouts and non-physical states only; falls continue so that recoveries are recorded."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    invalid_state = DoneTerm(func=g1_invalid_state, params={"maximum_height": None})


@configclass
class G129DofBfmEnvCfg(ManagerBasedRLEnvCfg):
    """G1 29-DoF driven by BFM-Zero latents at 50 Hz on 200 Hz MJWarp physics."""

    sim: SimulationCfg = SimulationCfg(dt=0.005, physics=G129DofPhysicsCfg())
    scene: G129DofSceneCfg = G129DofSceneCfg(num_envs=1024, env_spacing=2.5)
    actions: G129DofBfmActionsCfg = G129DofBfmActionsCfg()
    observations: G129DofBfmObservationsCfg = G129DofBfmObservationsCfg()
    events: G129DofBfmEventsCfg = G129DofBfmEventsCfg()
    rewards: G129DofBfmRewardsCfg = G129DofBfmRewardsCfg()
    terminations: G129DofBfmTerminationsCfg = G129DofBfmTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material


def configure_g1_29dof_robot(cfg: G1FlatEnvCfg) -> None:
    """Replace the stock 37-joint G1 with the 29-DoF robot in a velocity-tracking config.

    Joint-name dependent reward terms are remapped (no fingers; waist instead of ``torso_joint``) and the wrist links
    receive the same mass correction as the BFM-Zero collection environments. Physics settings are left unchanged.

    Args:
        cfg: Stock or NeRD G1 velocity config, modified in place.
    """
    cfg.scene.robot = G1_29DOF_BFM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", spawn=G1_29DOF_BFM_CFG.spawn.replace(activate_contact_sensors=True)
    )
    cfg.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
    )
    cfg.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="waist_.*_joint")
    cfg.rewards.joint_deviation_fingers = None
    cfg.events.wrist_mass = G129DofBfmEventsCfg().wrist_mass


def configure_g1_29dof_mjwarp_capacity(cfg: G1FlatEnvCfg) -> None:
    """Give the MJWarp preset of a stock G1 config constraint capacity for the full-collision 29-DoF asset.

    Args:
        cfg: Stock G1 velocity config whose ``sim.physics`` is the locomotion physics preset.
    """
    solver = cfg.sim.physics.newton_mjwarp.solver_cfg
    solver.njmax, solver.nconmax = G1_29DOF_MJWARP_PHYSICS.solver_cfg.njmax, G1_29DOF_MJWARP_PHYSICS.solver_cfg.nconmax


@configclass
class G129DofFlatEnvCfg(G1FlatEnvCfg):
    """Stock flat G1 velocity tracking with the 29-DoF robot; used to train the walking teacher."""

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_g1_29dof_robot(self)
        configure_g1_29dof_mjwarp_capacity(self)


@configclass
class G129DofPostureEnvCfg(G129DofFlatEnvCfg):
    """Stand in place while tracking commanded root height and torso roll and pitch.

    Held out from NeRD data collection; its teacher provides evaluation trajectories outside walking.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        velocity = self.commands.base_velocity
        velocity.ranges.lin_vel_x = velocity.ranges.lin_vel_y = velocity.ranges.ang_vel_z = (0.0, 0.0)
        velocity.heading_command = False
        velocity.ranges.heading = None
        velocity.rel_standing_envs = 1.0
        self.commands.posture = BaseHeightTorsoCommandCfg(resampling_time_range=(3.0, 5.0))
        self.observations.policy.posture_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "posture"}
        )
        self.rewards.track_height = RewTerm(
            func=track_base_height_exp,
            weight=2.0,
            params={"command_name": "posture", "std": 0.05, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.track_torso = RewTerm(
            func=track_torso_roll_pitch_exp,
            weight=2.0,
            params={
                "command_name": "posture",
                "std": 0.2,
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            },
        )
        self.rewards.feet_air_time = None
        self.rewards.joint_deviation_torso = None
        self.rewards.flat_orientation_l2.weight = -0.2


@configclass
class G129DofDatasetGenFlatEnvCfg(G1DatasetGenFlatEnvCfg):
    """Flat G1 dataset-generation config with the 29-DoF robot and the teacher's command and reset distribution.

    The 29-DoF teacher, trained on stock flat commands with the soft BFM-Zero gains, falls within seconds under the
    rich dataset commands (backward walking, 1 s resampling) and in about a fifth of episodes under joint-offset
    resets, both of which the stock 37-joint teacher tolerates. Stock flat commands and resets keep policy datasets on
    walking states.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_g1_29dof_robot(self)
        configure_g1_29dof_mjwarp_capacity(self)
        stock = G1FlatEnvCfg()
        self.commands.base_velocity = stock.commands.base_velocity
        self.events.reset_base = stock.events.reset_base
        self.events.reset_robot_joints = stock.events.reset_robot_joints


@configclass
class NerdG129DofFlatEnvCfg(NerdG1FlatEnvCfg):
    """Flat G1 velocity tracking with the 29-DoF robot on NeRD-backed physics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        configure_g1_29dof_robot(self)


@configclass
class G129DofBfmPerturbedEnvCfg(G129DofBfmEnvCfg):
    """BFM-Zero collection with lying starts, per-episode PD-gain scaling, action noise and root force pulses.

    The interval push term only runs in :meth:`~isaaclab.envs.ManagerBasedRLEnv.step`; frame-stepped dataset
    generation applies pushes in :class:`~isaaclab_neural.generate.samplers.g1_29dof_bfm.BfmLatentTrajectorySampler`.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.bfm.action_noise_std = 0.2
        self.events.reset_lying.params["fraction"] = 0.3
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )
        self.events.root_pushes = EventTerm(
            func=apply_root_wrench_pulses,
            mode="interval",
            interval_range_s=(self.decimation * self.sim.dt, self.decimation * self.sim.dt),
            params={"force_range": (0.0, 450.0), "duration_s": 0.1, "wait_range_s": (2.0, 5.0)},
        )
