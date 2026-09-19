# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka cube-lift environments for NeRD data generation and inference.

Both configurations derive from the core ``Isaac-Lift-Franka`` task (Menagerie
Panda, Lift-calibrated drives, 120 Hz physics with 30 Hz relative joint-position
control, finger contact sensors, ADR curriculum and reset bank). The official
Newton MJWarp checkpoint for that task therefore drives data collection directly.
"""

from __future__ import annotations

from isaaclab_newton.physics import NewtonCfg

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import FrankaLiftEnvCfg
from isaaclab_tasks.core.lift.lift_env_cfg import PhysicsCfg as LiftPhysicsCfg

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg

from .franka_scene import apply_franka_lift_nerd_scene, apply_franka_lift_nominal_physics
from .neural_env_wrapper import NerdManagerBasedRLEnv

DEFAULT_FRANKA_LIFT_NERD_MODEL_PATH: str | None = None


@configclass
class FrankaLiftCubeDatasetGenEnvCfg(FrankaLiftEnvCfg):
    """Core Franka Lift with a fixed cube and a static table for contact-aware NeRD data generation.

    Drives, actions, observations, rewards and resets are those of
    ``Isaac-Lift-Franka``. The object preset is pinned to the cube, the kinematic
    table becomes a static collider, domain randomization and the gravity ADR
    curriculum are removed, and CUDA graphs are disabled so the recorder can advance
    single physics frames.
    """

    finger_object_normal_force_sensors: tuple[str, ...] = ("panda_rightfinger_object_s", "panda_leftfinger_object_s")

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_franka_lift_nerd_scene(self)
        apply_franka_lift_nominal_physics(self)
        physics = self.sim.physics
        if not isinstance(physics, LiftPhysicsCfg):
            raise TypeError("Franka Lift dataset generation expects the core Lift physics presets.")
        # Configclass instances own deep copies of their defaults, so in-place edits stay local.
        physics.newton_mjwarp.use_cuda_graph = False
        physics.default = physics.newton_mjwarp


@configclass
class NerdFrankaLiftEnvCfg(FrankaLiftEnvCfg):
    """Core Franka Lift config using contact-aware NeRD physics."""

    # Core :class:`LiftPhysicsCfg` kept so Hydra ``presets=newton_mjwarp`` still
    # resolves after ``sim.physics`` is replaced.
    physics_presets: LiftPhysicsCfg | None = None

    finger_object_normal_force_sensors: tuple[str, ...] = ("panda_rightfinger_object_s", "panda_leftfinger_object_s")

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_franka_lift_nerd_scene(self)
        apply_franka_lift_nominal_physics(self)
        physics = self.sim.physics
        if not isinstance(physics, LiftPhysicsCfg) or not isinstance(physics.newton_mjwarp, NewtonCfg):
            raise TypeError("Franka Lift NeRD expects the core Lift Newton preset.")
        nerd_physics = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="TransformerNeuralSolver",
                neural_model_path=DEFAULT_FRANKA_LIFT_NERD_MODEL_PATH,
                num_states_history=10,
                num_contacts_per_env=64,
                contact_mode="newton_native",
                contact_representation="contact_tokens",
                max_contact_tokens=64,
                contact_packing_policy="body_round_robin_pair_atomic",
                min_contact_event_threshold=0.12,
                states_frame="body",
                anchor_frame_step="every",
                states_embedding_type="identical",
                prediction_type="relative",
                orientation_prediction_parameterization="quaternion",
            ),
            collision_cfg=physics.newton_mjwarp.collision_cfg,
            default_shape_cfg=physics.newton_mjwarp.default_shape_cfg,
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )
        physics.newton_mjwarp = nerd_physics
        physics.default = nerd_physics
        self.physics_presets = physics
        self.sim.physics = nerd_physics


class NerdFrankaLiftEnv(NerdManagerBasedRLEnv):
    """Franka cube-lift env that installs ``NerdNewtonCfg`` before simulation init."""

    robot_name = "Franka-Lift"
