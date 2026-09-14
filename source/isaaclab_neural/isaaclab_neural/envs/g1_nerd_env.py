# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registered Unitree G1 environments backed by the NeRD Newton solver."""

from __future__ import annotations

from copy import deepcopy

from isaaclab_newton.physics import NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.physics.nerd_solver_cfg import normalize_contact_mode

from ..contacts.geometry_mdp import GeometryFeetAirTime, geometry_illegal_contact
from .neural_env_wrapper import NerdManagerBasedRLEnv

DEFAULT_G1_FLAT_NERD_MODEL_PATH = "./pre-trained_models/G1/nn/final_model.pt"
DEFAULT_G1_ROUGH_NERD_MODEL_PATH = "./pre-trained_models/G1-Rough-Native/nn/final_model.pt"


def apply_g1_contact_mode_mdp(env_cfg) -> None:
    """Configure geometry-based contact MDP terms for G1 Newton-native contacts.

    G1 stock rewards use contact-force sensors (``feet_air_time``, ``feet_slide``,
    ``base_contact``). NeRD has no constraint-solver contact force, so those
    terms are remapped or disabled under ``newton_native``.
    """
    mode = env_cfg.sim.physics.solver_cfg.contact_mode
    env_cfg.scene.contact_forces = None
    # Force-sensor feet slide is unavailable without contact_forces.
    env_cfg.rewards.feet_slide = None
    if mode == "newton_native":
        env_cfg.rewards.feet_air_time = RewTerm(
            func=GeometryFeetAirTime,
            weight=0.25,
            params={
                "body_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
                "command_name": "base_velocity",
                "threshold": 0.4,
            },
        )
        env_cfg.terminations.base_contact = DoneTerm(
            func=geometry_illegal_contact,
            params={"body_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )
    else:
        env_cfg.rewards.feet_air_time = None
        env_cfg.terminations.base_contact = None


@configclass
class NerdG1FlatEnvCfg(G1FlatEnvCfg):
    """Manager-based G1 flat config using NeRD-backed Newton physics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        base_physics = resolve_presets(deepcopy(self.sim.physics))
        collision_cfg = deepcopy(getattr(base_physics, "collision_cfg", None))
        if collision_cfg is None:
            collision_cfg = NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000)
        default_shape_cfg = deepcopy(getattr(base_physics, "default_shape_cfg", None))
        if default_shape_cfg is None:
            default_shape_cfg = NewtonShapeCfg(margin=0.01)
        self.events.push_robot = None  # type: ignore[assignment]
        self.sim.physics = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="TransformerNeuralSolver",
                neural_model_path=DEFAULT_G1_FLAT_NERD_MODEL_PATH,
                num_states_history=10,
                num_contacts_per_env=64,
                max_contact_tokens=64,
                contact_mode="newton_native",
                contact_representation="contact_tokens",
                contact_filter="solver_active",
                contact_packing_policy="body_round_robin_pair_atomic",
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
        apply_g1_contact_mode_mdp(self)


class NerdG1FlatEnv(NerdManagerBasedRLEnv):
    """G1 flat env that installs ``NerdNewtonCfg`` before simulation initialization."""

    def _configure_solver_dependent_mdp(self, cfg) -> None:
        normalize_contact_mode(cfg.sim.physics.solver_cfg.contact_mode)


@configclass
class NerdG1RoughEnvCfg(G1RoughEnvCfg):
    """Manager-based G1 rough-terrain config using NeRD-backed Newton physics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        base_physics = resolve_presets(deepcopy(self.sim.physics))
        collision_cfg = deepcopy(getattr(base_physics, "collision_cfg", None))
        if collision_cfg is None:
            collision_cfg = NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000)
        default_shape_cfg = deepcopy(getattr(base_physics, "default_shape_cfg", None))
        if default_shape_cfg is None:
            default_shape_cfg = NewtonShapeCfg(margin=0.01)

        self.events.push_robot = None  # type: ignore[assignment]
        self.sim.physics = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="TransformerNeuralSolver",
                neural_model_path=DEFAULT_G1_ROUGH_NERD_MODEL_PATH,
                num_states_history=10,
                num_contacts_per_env=64,
                max_contact_tokens=64,
                contact_mode="newton_native",
                contact_representation="contact_tokens",
                contact_filter="solver_active",
                contact_packing_policy="body_round_robin_pair_atomic",
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
        apply_g1_contact_mode_mdp(self)


class NerdG1RoughEnv(NerdManagerBasedRLEnv):
    """G1 rough-terrain env that installs ``NerdNewtonCfg`` before simulation initialization."""

    def _configure_solver_dependent_mdp(self, cfg) -> None:
        normalize_contact_mode(cfg.sim.physics.solver_cfg.contact_mode)
