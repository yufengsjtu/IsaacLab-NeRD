# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registered Anymal-C flat environment backed by the NeRD Newton solver."""

from __future__ import annotations

from copy import deepcopy

from isaaclab_newton.physics import NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg

from ..contacts.geometry_mdp import GeometryFeetAirTime, geometry_illegal_contact, geometry_undesired_contacts
from .neural_env_wrapper import NerdManagerBasedRLEnv


DEFAULT_ANYMAL_C_FLAT_NERD_MODEL_PATH = "./pre-trained_models/Anymal-C/nn/final_model.pt"
DEFAULT_ANYMAL_C_ROUGH_NERD_MODEL_PATH = "./pre-trained_models/Anymal-C-Rough-Native/nn/final_model.pt"
ROOT_HEIGHT_MINIMUM = 0.4


def apply_contact_mode_mdp(env_cfg) -> None:
    """Configure geometry-based contact MDP terms for Newton-native contacts."""
    mode = env_cfg.sim.physics.solver_cfg.contact_mode
    # NeRD has no constraint-solver contact force. Do not initialize the stock
    # force sensor merely to infer binary contact state.
    env_cfg.scene.contact_forces = None
    if mode == "newton_native":
        env_cfg.rewards.feet_air_time = RewTerm(
            func=GeometryFeetAirTime,
            weight=0.5,
            params={
                "body_cfg": SceneEntityCfg("robot", body_names=".*FOOT"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )
        env_cfg.rewards.undesired_contacts = RewTerm(
            func=geometry_undesired_contacts,
            weight=-1.0,
            params={"body_cfg": SceneEntityCfg("robot", body_names=".*THIGH")},
        )
        env_cfg.terminations.base_contact = DoneTerm(
            func=geometry_illegal_contact,
            params={"body_cfg": SceneEntityCfg("robot", body_names="base")},
        )
    else:
        env_cfg.rewards.feet_air_time = None
        env_cfg.rewards.undesired_contacts = None
        env_cfg.terminations.base_contact = None


@configclass
class FlatObservationsCfg:
    """Policy observations for flat Anymal-C NeRD (stock 48-D layout)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # root_height = ObsTerm(
        #     func=base_mdp.base_pos_z,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class NerdAnymalCFlatEnvCfg(AnymalCFlatEnvCfg):
    """Manager-based AnymalCFlat config using NeRD-backed Newton physics.

    ``feet_air_time`` / ``undesired_contacts`` only when ``contact_mode=newton_native``.
    """

    observations: FlatObservationsCfg = FlatObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # self.terminations.base_contact = None  # type: ignore[assignment]
        self.events.push_robot = None  # type: ignore[assignment]

        # self.terminations.low_root_height = DoneTerm(  # type: ignore[attr-defined]
        #     func=base_mdp.root_height_below_minimum,
        #     params={
        #         "minimum_height": ROOT_HEIGHT_MINIMUM,
        #         "asset_cfg": SceneEntityCfg("robot"),
        #     },
        # )
        # Validated flat NeRD PPO (2026-08-07_15-38-10): stronger tracking vs stock.
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.sim.physics = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="TransformerNeuralSolver",
                neural_model_path=DEFAULT_ANYMAL_C_FLAT_NERD_MODEL_PATH,
                num_states_history=10,
                num_contacts_per_env=0,
                contact_mode="fixed_ground",
                contact_packing_policy="penetration_priority",
                min_contact_event_threshold=0.12,
                states_frame="body",
                anchor_frame_step="every",
                states_embedding_type="identical",
                prediction_type="relative",
                orientation_prediction_parameterization="quaternion",
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )
        apply_contact_mode_mdp(self)


class NerdAnymalCFlatEnv(NerdManagerBasedRLEnv):
    """AnymalCFlat env that installs ``NerdNewtonCfg`` before IsaacLab initializes simulation."""


@configclass
class NerdAnymalCRoughEnvCfg(AnymalCRoughEnvCfg):
    """Manager-based AnymalC rough-terrain config using NeRD-backed Newton physics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        base_physics = self.sim.physics
        collision_cfg = deepcopy(getattr(base_physics, "collision_cfg", None))
        if collision_cfg is None:
            collision_cfg = NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000)
        default_shape_cfg = deepcopy(getattr(base_physics, "default_shape_cfg", None))
        if default_shape_cfg is None:
            default_shape_cfg = NewtonShapeCfg(margin=0.01)

        self.terminations.base_contact = None  # type: ignore[assignment]
        self.events.push_robot = None  # type: ignore[assignment]

        self.sim.physics = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="TransformerNeuralSolver",
                neural_model_path=DEFAULT_ANYMAL_C_ROUGH_NERD_MODEL_PATH,
                num_states_history=10,
                num_contacts_per_env=64,
                contact_mode="newton_native",
                contact_packing_policy="penetration_priority",
                min_contact_event_threshold=0.12,
                states_frame="body",
                anchor_frame_step="every",
                states_embedding_type="identical",
                prediction_type="relative",
                orientation_prediction_parameterization="quaternion",
            ),
            collision_cfg=collision_cfg,
            default_shape_cfg=default_shape_cfg,
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )
        apply_contact_mode_mdp(self)


class NerdAnymalCRoughEnv(NerdManagerBasedRLEnv):
    """AnymalC rough-terrain env that installs ``NerdNewtonCfg`` before simulation initialization."""
