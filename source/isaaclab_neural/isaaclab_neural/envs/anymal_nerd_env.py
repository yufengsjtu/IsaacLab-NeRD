# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registered Anymal-C flat environment backed by the NeRD Newton solver."""

from __future__ import annotations

from copy import deepcopy

from isaaclab_newton.physics import NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg

from .neural_env_wrapper import NerdManagerBasedRLEnv


DEFAULT_ANYMAL_C_FLAT_NERD_MODEL_PATH = "./pre-trained_models/Anymal-C/nn/final_model.pt"
DEFAULT_ANYMAL_C_ROUGH_NERD_MODEL_PATH = "./pre-trained_models/Anymal-C-Rough-Native/nn/final_model.pt"
ROOT_HEIGHT_MINIMUM = 0.4


@configclass
class NerdAnymalCFlatEnvCfg(AnymalCFlatEnvCfg):
    """Manager-based AnymalCFlat config using NeRD-backed Newton physics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Disable contact-sensor related reward/termination terms (not working with NeRD solver)
        self.terminations.base_contact = None  # type: ignore[assignment]
        self.rewards.feet_air_time = None  # type: ignore[assignment]
        self.rewards.undesired_contacts = None  # type: ignore[assignment]
        # Disable random pushing event
        self.events.push_robot = None  # type: ignore[assignment]

        # Add root-height termination
        self.terminations.low_root_height = DoneTerm(  # type: ignore[attr-defined]
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

        # Disable contact-sensor related reward/termination terms (not working with NeRD solver)
        self.terminations.base_contact = None  # type: ignore[assignment]
        self.rewards.feet_air_time = None  # type: ignore[assignment]
        self.rewards.undesired_contacts = None  # type: ignore[assignment]
        # Disable random pushing event
        self.events.push_robot = None  # type: ignore[assignment]

        # Add root-height termination
        self.terminations.low_root_height = DoneTerm(  # type: ignore[attr-defined]
            func=base_mdp.root_height_below_minimum,
            params={
                "minimum_height": ROOT_HEIGHT_MINIMUM,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

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


class NerdAnymalCRoughEnv(NerdManagerBasedRLEnv):
    """AnymalC rough-terrain env that installs ``NerdNewtonCfg`` before simulation initialization."""
