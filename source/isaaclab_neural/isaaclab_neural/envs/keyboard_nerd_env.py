# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO101 keyboard NeRD environments (fixed or randomized keyboard scene) with robot normal forces."""

from isaaclab.sim import UsdFileCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.keyboard.so101_env_cfg import KeyboardAssetCfg, PhysicsCfg, SO101KeyboardEnvCfg

from isaaclab_neural.generate.keyboard_dataset import apply_keyboard_scene
from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg

from .neural_env_wrapper import NerdManagerBasedRLEnv


@configclass
class NerdSO101KeyboardEnvCfg(SO101KeyboardEnvCfg):
    """Keep the native task contract; by default fix keyboard variant 0 and its authored pose.

    One learned step predicts one 0.01 s physics frame. The native teacher uses
    two 0.005 s internal substeps; control still runs every four physics frames.
    An explicit compatible checkpoint is required outside deferred training.
    """

    robot_normal_force_sensor: str = "robot_contact"
    keyboard_scene: str = "fixed"
    """``"fixed"`` (variant 0, authored pose) or ``"randomized"`` (all variants, random root-pose resets)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        physics = self.sim.physics
        if isinstance(physics, NerdNewtonCfg):
            return
        if not isinstance(physics, PhysicsCfg):
            raise TypeError("SO101 Keyboard NeRD expects the native keyboard physics presets.")
        keyboard, robot_spawn = self.scene.keyboard, self.scene.robot.spawn
        if not isinstance(keyboard, KeyboardAssetCfg) or not isinstance(robot_spawn, UsdFileCfg):
            raise TypeError("SO101 Keyboard NeRD requires the native keyboard presets and USD robot.")
        if not isinstance(robot_spawn.variants, dict):
            raise TypeError("SO101 robot requires a Physics variant mapping.")
        self.scene.keyboard = keyboard.newton_mjwarp.copy()
        robot_spawn.variants["Physics"] = "physics"
        apply_keyboard_scene(self, self.keyboard_scene, variant=0)
        teacher = physics.newton_mjwarp
        nerd = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="TransformerNeuralSolver",
                num_states_history=4,
                states_frame="body",
                states_embedding_type="identical",
                prediction_type="relative",
                contact_representation="contact_tokens",
                contact_token_frame="owner",
                num_contacts_per_env=256,
                max_contact_tokens=256,
                contact_packing_policy="body_round_robin_pair_atomic",
            ),
            collision_cfg=teacher.collision_cfg,
            default_shape_cfg=teacher.default_shape_cfg,
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )
        self.sim.physics = nerd
        self.scene.num_envs = 8


@configclass
class NerdSO101KeyboardRandomizedEnvCfg(NerdSO101KeyboardEnvCfg):
    """The original randomized scene: every keyboard variant (round-robin over worlds) and random root poses.

    The model receives no keyboard pose or variant input; keyboard geometry reaches it only through contacts.
    """

    keyboard_scene: str = "randomized"


class NerdSO101KeyboardEnv(NerdManagerBasedRLEnv):
    """SO101 keyboard task using NeRD state and normal-force prediction."""

    @property
    def robot_name(self) -> str:
        """Dataset task identity for the fixed keyboard domain."""
        return "SO101-Keyboard-Fixed"

    @property
    def frame_dt(self) -> float:
        """Duration [s] of one supervised dynamics transition, not a control interval."""
        return self.physics_dt
