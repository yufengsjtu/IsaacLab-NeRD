# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Allegro in-hand cube reorientation environments for NeRD policy, data generation and inference.

All configurations derive from the core manager-based ``Isaac-Reorient-Cube-Allegro``
task (fixed-base Allegro hand, free DexCube, 120 Hz physics with two MJWarp substeps
and 30 Hz joint-position control). NeRD predicts the hand and the cube together, so
contacts come from Newton's collision pipeline instead of MuJoCo's internal one; the
teacher policy is trained on the same contact pipeline it is later recorded with.
"""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_common import PhysicsCfg as AllegroPhysicsCfg
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_manager_env_cfg import AllegroHandManagerEnvCfg

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg

from .neural_env_wrapper import NerdManagerBasedRLEnv

DEFAULT_ALLEGRO_REORIENT_NERD_MODEL_PATH: str | None = None

ALLEGRO_REORIENT_NUM_CONTACTS: int = 128
"""Native contact-token capacity per world for the hand, the cube and hand self-contacts."""

ALLEGRO_REORIENT_NJMAX: int = 256
"""MJWarp constraint-row capacity per world with Newton contacts.

The core ``njmax=80`` overflowed (91 rows requested) once Newton's collision
pipeline supplied the contacts, which silently drops constraints.
"""

ALLEGRO_REORIENT_NCONMAX: int = 128
"""MJWarp contact capacity per world with Newton contacts."""


def apply_allegro_reorient_newton_contacts(physics: AllegroPhysicsCfg) -> NewtonCfg:
    """Switch the Allegro MJWarp preset to Newton's collision pipeline and select it.

    Solver settings and substeps stay those of the core task; constraint capacities
    are raised to :data:`ALLEGRO_REORIENT_NJMAX` and :data:`ALLEGRO_REORIENT_NCONMAX`.

    Args:
        physics: The core Allegro physics preset, edited in place.

    Returns:
        The selected Newton MJWarp configuration.
    """
    newton_cfg = physics.newton_mjwarp
    if not isinstance(newton_cfg, NewtonCfg) or not isinstance(newton_cfg.solver_cfg, MJWarpSolverCfg):
        raise TypeError("Allegro reorientation NeRD expects the core Allegro Newton MJWarp preset.")
    newton_cfg.solver_cfg.use_mujoco_contacts = False
    newton_cfg.solver_cfg.njmax = max(newton_cfg.solver_cfg.njmax, ALLEGRO_REORIENT_NJMAX)
    newton_cfg.solver_cfg.nconmax = max(newton_cfg.solver_cfg.nconmax, ALLEGRO_REORIENT_NCONMAX)
    if newton_cfg.collision_cfg is None:
        newton_cfg.collision_cfg = NewtonCollisionPipelineCfg()
    if newton_cfg.default_shape_cfg is None:
        newton_cfg.default_shape_cfg = NewtonShapeCfg()
    physics.default = newton_cfg
    return newton_cfg


@configclass
class AllegroReorientPolicyEnvCfg(AllegroHandManagerEnvCfg):
    """Core Allegro reorientation on Newton contacts, used to train the data-collection policy.

    Actions, observations, rewards, commands, resets and terminations are those of
    ``Isaac-Reorient-Cube-Allegro`` without domain randomization (the core default).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.sim.physics, AllegroPhysicsCfg):
            raise TypeError("Allegro reorientation NeRD expects the core Allegro physics presets.")
        apply_allegro_reorient_newton_contacts(self.sim.physics)


@configclass
class AllegroReorientDatasetGenEnvCfg(AllegroReorientPolicyEnvCfg):
    """Policy environment with CUDA graphs disabled so the recorder can advance single physics frames."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.physics.newton_mjwarp.use_cuda_graph = False


@configclass
class NerdAllegroReorientEnvCfg(AllegroReorientPolicyEnvCfg):
    """Allegro reorientation with the hand and the cube advanced by contact-aware NeRD."""

    # Core preset kept so Hydra ``presets=newton_mjwarp`` still resolves after ``sim.physics`` is replaced.
    physics_presets: AllegroPhysicsCfg | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        physics = self.sim.physics
        teacher = physics.newton_mjwarp
        nerd_physics = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="TransformerNeuralSolver",
                neural_model_path=DEFAULT_ALLEGRO_REORIENT_NERD_MODEL_PATH,
                num_states_history=10,
                num_contacts_per_env=ALLEGRO_REORIENT_NUM_CONTACTS,
                contact_mode="newton_native",
                contact_representation="contact_tokens",
                contact_filter="solver_active",
                max_contact_tokens=ALLEGRO_REORIENT_NUM_CONTACTS,
                contact_packing_policy="body_round_robin_pair_atomic",
                states_frame="body",
                anchor_frame_step="every",
                states_embedding_type="identical",
                prediction_type="relative",
                orientation_prediction_parameterization="quaternion",
            ),
            collision_cfg=teacher.collision_cfg,
            default_shape_cfg=teacher.default_shape_cfg,
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )
        physics.newton_mjwarp = nerd_physics
        physics.default = nerd_physics
        self.physics_presets = physics
        self.sim.physics = nerd_physics


class NerdAllegroReorientEnv(NerdManagerBasedRLEnv):
    """Allegro reorientation env that installs ``NerdNewtonCfg`` before simulation init."""

    robot_name = "Allegro-Reorient"
