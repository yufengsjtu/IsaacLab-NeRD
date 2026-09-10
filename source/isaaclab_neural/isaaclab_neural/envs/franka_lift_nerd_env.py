# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka cube-lift environments for NeRD data generation and inference."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg

from .franka_scene import apply_franka_lift_nerd_scene
from .neural_env_wrapper import NerdManagerBasedRLEnv

DEFAULT_FRANKA_LIFT_NERD_MODEL_PATH: str | None = None


@configclass
class FrankaLiftNeRDPhysicsCfg(PresetCfg):
    """Physics presets for Franka lift NeRD dataset generation.

    Stock :class:`~isaaclab_tasks.contrib.lift.lift_env_cfg.LiftEnvCfg`
    is PhysX-only. Dataset generation always selects ``presets=newton_mjwarp``.
    """

    default: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=16 * 1024,
        friction_correlation_distance=0.00625,
    )
    physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=16 * 1024,
        friction_correlation_distance=0.00625,
    )
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=90,
            nconmax=100,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        default_shape_cfg=NewtonShapeCfg(margin=0.01),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=False,
    )


@configclass
class FrankaLiftCubeDatasetGenEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka cube-lift config for contact-aware NeRD dataset generation.

    The cube stays a free rigid object. The table is a static cuboid collider so
    Newton does not add a second unactuated table articulation. Random actions
    are not a substitute for successful grasp coverage.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_franka_lift_nerd_scene(self)
        self.sim.physics = FrankaLiftNeRDPhysicsCfg()


@configclass
class NerdFrankaLiftEnvCfg(FrankaCubeLiftEnvCfg):
    """Manager-based Franka cube-lift config using contact-aware NeRD physics."""

    # Dataset-gen :class:`FrankaLiftNeRDPhysicsCfg` kept so Hydra
    # ``presets=newton_mjwarp`` still resolves after ``sim.physics`` is replaced.
    physics_presets: FrankaLiftNeRDPhysicsCfg | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_franka_lift_nerd_scene(self)
        nerd_physics = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="TransformerNeuralSolver",
                neural_model_path=DEFAULT_FRANKA_LIFT_NERD_MODEL_PATH,
                num_states_history=10,
                num_contacts_per_env=64,
                contact_mode="newton_native",
                contact_packing_policy="penetration_priority",
                min_contact_event_threshold=0.12,
                states_frame="world",
                anchor_frame_step="every",
                states_embedding_type="identical",
                prediction_type="relative",
                orientation_prediction_parameterization="quaternion",
            ),
            collision_cfg=NewtonCollisionPipelineCfg(),
            default_shape_cfg=NewtonShapeCfg(margin=0.01),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )
        physics_presets = FrankaLiftNeRDPhysicsCfg()
        physics_presets.newton_mjwarp = nerd_physics
        self.physics_presets = physics_presets
        self.sim.physics = nerd_physics


class NerdFrankaLiftEnv(NerdManagerBasedRLEnv):
    """Franka cube-lift env that installs ``NerdNewtonCfg`` before simulation init."""

    robot_name = "Franka-Lift"
