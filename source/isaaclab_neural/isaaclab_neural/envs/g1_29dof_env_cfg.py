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

from isaaclab_neural.controllers.bfm_zero import BfmZeroLatentActionCfg
from isaaclab_neural.envs.g1_nerd_env import g1_invalid_state
from isaaclab_neural.envs.mdp_g1_29dof import apply_root_wrench_pulses, reset_root_lying
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

    sim: SimulationCfg = SimulationCfg(dt=0.005, physics=G1_29DOF_MJWARP_PHYSICS)
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


@configclass
class G129DofBfmPushEnvCfg(G129DofBfmEnvCfg):
    """BFM-Zero collection with random root force pulses and lying starts."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.reset_lying.params["fraction"] = 0.3
        self.events.root_pushes = EventTerm(
            func=apply_root_wrench_pulses,
            mode="interval",
            interval_range_s=(self.decimation * self.sim.dt, self.decimation * self.sim.dt),
            params={"force_range": (0.0, 450.0), "duration_s": 0.1, "wait_range_s": (2.0, 5.0)},
        )
