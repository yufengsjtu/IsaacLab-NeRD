# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config-level tests for Franka Reach and Lift NeRD tasks."""

from __future__ import annotations

import gymnasium as gym
import isaaclab_neural.envs  # noqa: F401
import numpy as np
from isaaclab_neural.envs.franka_lift_nerd_env import FrankaLiftCubeDatasetGenEnvCfg, NerdFrankaLiftEnvCfg
from isaaclab_neural.envs.franka_reach_nerd_env import (
    FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S,
    FRANKA_REACH_DATASET_INCREMENT_SCALE,
    FRANKA_REACH_DATASET_SOFT_JOINT_POS_LIMIT_FACTOR,
    FrankaReachDatasetGenEnvCfg,
    NerdFrankaReachEnvCfg,
)
from isaaclab_neural.envs.incremental_joint_position import IncrementalJointPositionActionCfg
from isaaclab_neural.physics import NerdNewtonCfg
from isaaclab_neural.utils.commons import JOINT_F_LIM, JOINT_Q_MAX, JOINT_Q_MIN, JOINT_QD_LIM
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

from isaaclab_tasks.utils.hydra import collect_presets


def test_franka_reach_nerd_is_contact_free_and_tableless() -> None:
    cfg = NerdFrankaReachEnvCfg()

    assert cfg.scene.table is None
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is False
    assert cfg.scene.robot.spawn.usd_path.endswith("Robots/FrankaEmika/Legacy/panda_instanceable.usd")
    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    assert cfg.sim.physics.solver_cfg.contact_mode == "newton_native"
    assert cfg.sim.physics.solver_cfg.num_contacts_per_env == 0
    assert cfg.sim.physics.solver_cfg.states_frame == "world"
    preset_names = {name for fields in collect_presets(cfg).values() for name in fields}
    assert "newton_mjwarp" in preset_names


def test_franka_reach_dataset_gen_drops_table_articulation() -> None:
    cfg = FrankaReachDatasetGenEnvCfg()

    assert cfg.scene.table is None
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is False
    assert cfg.scene.robot.spawn.usd_path.endswith("Robots/FrankaEmika/Legacy/panda_instanceable.usd")
    assert hasattr(cfg.sim.physics, "newton_mjwarp")
    assert cfg.commands.ee_pose.debug_vis is False
    assert isinstance(cfg.actions.arm_action, IncrementalJointPositionActionCfg)
    assert cfg.actions.arm_action.scale == FRANKA_REACH_DATASET_INCREMENT_SCALE
    assert cfg.actions.arm_action.scale["panda_joint[1-4]"] == 87.0 / 80.0
    assert cfg.actions.arm_action.scale["panda_joint[5-7]"] == 12.0 / 80.0
    assert cfg.scene.robot.soft_joint_pos_limit_factor == FRANKA_REACH_DATASET_SOFT_JOINT_POS_LIMIT_FACTOR
    assert FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S == 1.5
    assert cfg.commands.ee_pose.resampling_time_range == (
        FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S,
        FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S,
    )
    cfg.restore_reach_rl_arm_action()
    assert isinstance(cfg.actions.arm_action, JointPositionActionCfg)
    assert cfg.actions.arm_action.scale == 0.5
    assert cfg.actions.arm_action.use_default_offset is True


def test_franka_reach_nerd_keeps_rl_joint_position_action() -> None:
    cfg = NerdFrankaReachEnvCfg()

    assert isinstance(cfg.actions.arm_action, JointPositionActionCfg)
    assert cfg.actions.arm_action.scale == 0.5
    assert cfg.actions.arm_action.use_default_offset is True
    assert cfg.scene.robot.soft_joint_pos_limit_factor == 1.0
    assert cfg.commands.ee_pose.resampling_time_range == (4.0, 4.0)


def test_franka_lift_nerd_keeps_contacts_and_static_table() -> None:
    cfg = NerdFrankaLiftEnvCfg()

    assert cfg.scene.table is not None
    assert cfg.scene.robot.spawn.usd_path.endswith("Robots/FrankaEmika/Legacy/panda_instanceable.usd")
    assert cfg.scene.table.spawn.size == (0.9, 1.3, 0.05)
    assert cfg.scene.table.spawn.rigid_props is None
    assert cfg.scene.object is not None
    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    assert cfg.sim.physics.solver_cfg.num_contacts_per_env == 64
    assert cfg.sim.physics.solver_cfg.states_frame == "world"
    preset_names = {name for fields in collect_presets(cfg).values() for name in fields}
    assert "newton_mjwarp" in preset_names


def test_franka_lift_dataset_gen_exposes_newton_mjwarp() -> None:
    cfg = FrankaLiftCubeDatasetGenEnvCfg()

    assert cfg.scene.table.spawn.size == (0.9, 1.3, 0.05)
    assert isinstance(cfg.sim.physics.newton_mjwarp, NewtonCfg)
    assert cfg.sim.physics.newton_mjwarp.solver_cfg.nconmax == 100
    assert cfg.sim.physics.newton_mjwarp.use_cuda_graph is False


def test_franka_sampling_limits_match_expected_dof() -> None:
    assert JOINT_Q_MIN["Franka"].shape == (9,)
    assert JOINT_Q_MAX["Franka"].shape == (9,)
    assert np.asarray(JOINT_QD_LIM["Franka"]).shape == (9,)
    assert JOINT_F_LIM["Franka"].shape == (9,)
    assert JOINT_Q_MIN["Franka-Lift"].shape == (16,)
    assert JOINT_Q_MAX["Franka-Lift"].shape == (16,)
    assert JOINT_QD_LIM["Franka-Lift"].shape == (15,)
    assert JOINT_F_LIM["Franka-Lift"].shape == (15,)
    np.testing.assert_array_equal(JOINT_F_LIM["Franka"][-2:], np.zeros(2))


def test_franka_reach_hold_is_twenty_physics_frames() -> None:
    from isaaclab_neural.envs.franka_reach_nerd_env import FRANKA_REACH_DATASET_ACTION_HOLD_FRAMES
    from isaaclab_neural.generate.sampler import resolve_action_resample_period

    assert FRANKA_REACH_DATASET_ACTION_HOLD_FRAMES == 20
    assert (
        resolve_action_resample_period(
            "frame", env_decimation=2, action_hold_frames=FRANKA_REACH_DATASET_ACTION_HOLD_FRAMES
        )
        == 20
    )


def test_franka_gym_ids_are_registered() -> None:
    for env_id in (
        "Isaac-Reach-Franka-NeRD-v0",
        "Isaac-Reach-Franka-Dataset-Gen-v0",
        "Isaac-Lift-Cube-Franka-NeRD-v0",
        "Isaac-Lift-Cube-Franka-Dataset-Gen-v0",
    ):
        assert env_id in gym.registry
