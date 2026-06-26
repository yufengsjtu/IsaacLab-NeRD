# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registered Anymal-C flat environment backed by the NeRD Newton solver."""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg

from .neural_env_wrapper import NerdManagerBasedRLEnv


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


DEFAULT_ANYMAL_C_FLAT_NERD_MODEL_PATH = "/home/rowany/workspace/pre-trained_models/Anymal-C/nn/final_model.pt"
ROOT_HEIGHT_MINIMUM = 0.4


@configclass
class NerdAnymalCFlatEnvCfg(AnymalCFlatEnvCfg):
    """Manager-based AnymalCFlat config using NeRD-backed Newton physics."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Disable contact-sensor related reward/termination terms (not working with NeRD solver)
        self.terminations.base_contact = None
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        # Disable random pushing event
        self.events.push_robot = None

        # Add root-height termination
        self.terminations.low_root_height = DoneTerm(
            func=base_mdp.root_height_below_minimum,
            params={
                "minimum_height": ROOT_HEIGHT_MINIMUM,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.sim.physics = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="TransformerNeuralSolver",
                neural_model_path=DEFAULT_ANYMAL_C_FLAT_NERD_MODEL_PATH,
                num_states_history=10,
                num_contacts_per_env=0,
                contact_mode="fixed_ground",
                contact_packing_policy="stable_index",
                min_contact_event_threshold=0.12,
                validate_contact_fingerprint=False,
                states_frame="body",
                anchor_frame_step="every",
                states_embedding_type="identical",
                prediction_type="relative",
                orientation_prediction_parameterization="quaternion",
                use_cuda_graph=False,
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )


class NerdAnymalCFlatEnv(NerdManagerBasedRLEnv):
    """AnymalCFlat env that installs ``NerdNewtonCfg`` before IsaacLab initializes simulation."""
