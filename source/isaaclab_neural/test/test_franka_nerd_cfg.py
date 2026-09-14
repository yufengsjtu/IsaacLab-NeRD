# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config-level tests for Franka Reach and Lift NeRD tasks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import gymnasium as gym
import isaaclab_neural.envs  # noqa: F401
import numpy as np
import pytest
import torch
from isaaclab_neural.envs.franka_lift_nerd_env import FrankaLiftCubeDatasetGenEnvCfg, NerdFrankaLiftEnvCfg
from isaaclab_neural.envs.franka_reach_nerd_env import (
    FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S,
    FRANKA_REACH_DATASET_SOFT_JOINT_POS_LIMIT_FACTOR,
    FrankaReachDatasetGenEnvCfg,
    NerdFrankaReachEnvCfg,
)
from isaaclab_neural.envs.franka_scene import FRANKA_PANDA_CFG as NERD_FRANKA_PANDA_CFG
from isaaclab_neural.envs.franka_scene import apply_franka_lift_nerd_scene
from isaaclab_neural.envs.incremental_joint_position import IncrementalJointPositionActionCfg
from isaaclab_neural.physics import NerdNewtonCfg
from isaaclab_neural.utils.commons import JOINT_F_LIM, JOINT_Q_MAX, JOINT_Q_MIN, JOINT_QD_LIM
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.string import resolve_matching_names

from isaaclab_tasks.contrib.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
from isaaclab_tasks.core.reach.config.franka.franka_reach_env_cfg import FrankaReachEnvCfg
from isaaclab_tasks.core.reach.reach_env_cfg import EventCfg as ReachEventCfg
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets


def test_nerd_inlined_panda_matches_shared_legacy_asset() -> None:
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    inlined = NERD_FRANKA_PANDA_CFG.to_dict()
    shared = FRANKA_PANDA_CFG.to_dict()
    assert inlined["actuators"] == shared["actuators"]
    assert inlined["init_state"] == shared["init_state"]
    assert inlined["spawn"]["usd_path"] == shared["spawn"]["usd_path"]
    assert inlined["soft_joint_pos_limit_factor"] == shared["soft_joint_pos_limit_factor"]


@pytest.mark.parametrize("cfg_type", [FrankaReachEnvCfg, FrankaReachDatasetGenEnvCfg, NerdFrankaReachEnvCfg])
def test_franka_reach_pipeline_defaults_to_legacy_at_60_hz(cfg_type) -> None:
    cfg = resolve_presets(cfg_type(), {"newton_mjwarp"})
    assert "/Assets/Isaac/6.0/" in cfg.scene.robot.spawn.usd_path
    assert cfg.scene.robot.spawn.usd_path.endswith("/Legacy/panda_instanceable.usd")
    assert set(cfg.scene.robot.actuators) == {"panda_shoulder", "panda_forearm", "panda_hand"}
    assert cfg.sim.dt == 1.0 / 60.0
    assert cfg.decimation == 2
    assert cfg.sim.physics.num_substeps == 1
    assert cfg.rewards.success is None
    assert cfg.terminations.success is None


def test_franka_reach_nerd_is_contact_free_and_tableless() -> None:
    cfg = NerdFrankaReachEnvCfg()

    assert cfg.scene.table is None
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is False
    standard = resolve_presets(FrankaReachEnvCfg(), {"newton_mjwarp"})
    resolve_presets(cfg, {"newton_mjwarp"})
    standard.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    assert cfg.scene.robot.to_dict() == standard.scene.robot.to_dict()
    for field in ("observations", "actions", "events", "commands", "rewards", "terminations", "curriculum"):
        assert getattr(cfg, field).to_dict() == getattr(standard, field).to_dict(), field
    assert cfg.sim.dt == standard.sim.dt
    assert cfg.decimation == standard.decimation
    assert cfg.episode_length_s == standard.episode_length_s
    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    assert cfg.sim.physics.solver_cfg.contact_mode == "newton_native"
    assert cfg.sim.physics.solver_cfg.num_contacts_per_env == 0
    assert cfg.sim.physics.solver_cfg.states_frame == "world"
    preset_names = {name for fields in collect_presets(NerdFrankaReachEnvCfg()).values() for name in fields}
    assert "newton_mjwarp" in preset_names


def test_franka_reach_dataset_gen_drops_table_articulation() -> None:
    cfg = FrankaReachDatasetGenEnvCfg()

    assert cfg.scene.table is None
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is False
    standard = FrankaReachEnvCfg()
    standard.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    standard.scene.robot.soft_joint_pos_limit_factor = FRANKA_REACH_DATASET_SOFT_JOINT_POS_LIMIT_FACTOR
    assert cfg.scene.robot.to_dict() == standard.scene.robot.to_dict()
    assert hasattr(cfg.sim.physics, "newton_mjwarp")
    assert cfg.commands.ee_pose.debug_vis is False
    assert isinstance(cfg.actions.arm_action, IncrementalJointPositionActionCfg)
    assert cfg.actions.arm_action.scale is None
    assert cfg.scene.robot.soft_joint_pos_limit_factor == FRANKA_REACH_DATASET_SOFT_JOINT_POS_LIMIT_FACTOR == 0.6
    assert FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S == 1.5
    assert cfg.commands.ee_pose.resampling_time_range == (
        FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S,
        FRANKA_REACH_DATASET_COMMAND_RESAMPLE_S,
    )
    assert cfg.events.reset_robot_joints.func is mdp.reset_joints_within_limits_range
    cfg.restore_reach_rl_arm_action()
    assert isinstance(cfg.actions.arm_action, JointPositionActionCfg)
    assert cfg.actions.arm_action.scale == 0.5
    assert cfg.actions.arm_action.use_default_offset is True
    assert cfg.events.reset_robot_joints.to_dict() == ReachEventCfg().reset_robot_joints.to_dict()
    assert FrankaReachDatasetGenEnvCfg().events.reset_robot_joints.func is mdp.reset_joints_within_limits_range


def test_franka_random_reset_covers_arm_limits_and_preserves_partial_reset_scope(mocker) -> None:
    cfg = FrankaReachDatasetGenEnvCfg()
    num_envs = 512
    joint_names = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
    limits = torch.tensor([[-2.0, 2.0]] * 7 + [[0.004, 0.036]] * 2).repeat(num_envs, 1, 1)
    default_pos = torch.tensor([0.0] * 7 + [0.04] * 2).repeat(num_envs, 1)
    asset = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=default_pos),
            default_joint_vel=SimpleNamespace(torch=torch.zeros_like(default_pos)),
            soft_joint_pos_limits=SimpleNamespace(torch=limits),
            soft_joint_vel_limits=SimpleNamespace(torch=torch.ones_like(default_pos)),
        ),
        find_joints=lambda names: resolve_matching_names(names, joint_names),
        write_joint_position_to_sim_index=mocker.Mock(),
        write_joint_velocity_to_sim_index=mocker.Mock(),
    )
    env = SimpleNamespace(scene={"robot": asset}, num_envs=num_envs, device="cpu")
    env_ids = torch.arange(1, num_envs, 2)
    reset_cfg = cfg.events.reset_robot_joints
    with torch.random.fork_rng():
        torch.manual_seed(0)
        term = reset_cfg.func(reset_cfg, env)
        term(env, env_ids, **reset_cfg.params)
    position_call = asset.write_joint_position_to_sim_index.call_args.kwargs
    velocity_call = asset.write_joint_velocity_to_sim_index.call_args.kwargs
    position = position_call["position"]
    assert position.shape == (len(env_ids), len(joint_names))
    torch.testing.assert_close(position_call["env_ids"], env_ids)
    torch.testing.assert_close(velocity_call["env_ids"], env_ids)
    assert torch.all(position >= limits[env_ids, :, 0])
    assert torch.all(position <= limits[env_ids, :, 1])
    normalized = (position[:, :7] - limits[env_ids, :7, 0]) / (limits[env_ids, :7, 1] - limits[env_ids, :7, 0])
    assert torch.all(normalized.amin(dim=0) < 0.1)
    assert torch.all(normalized.amax(dim=0) > 0.9)
    expected_fingers = default_pos[env_ids, 7:].clamp(limits[env_ids, 7:, 0], limits[env_ids, 7:, 1])
    torch.testing.assert_close(position[:, 7:], expected_fingers)
    torch.testing.assert_close(velocity_call["velocity"], torch.zeros_like(position))


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
    standard = FrankaCubeLiftEnvCfg()
    for scene_field in ("robot", "object", "ee_frame", "plane", "light"):
        assert getattr(cfg.scene, scene_field).to_dict() == getattr(standard.scene, scene_field).to_dict()
    for field in ("observations", "actions", "events", "commands", "rewards", "terminations", "curriculum"):
        assert getattr(cfg, field).to_dict() == getattr(standard, field).to_dict(), field
    assert cfg.sim.dt == standard.sim.dt
    assert cfg.decimation == standard.decimation
    assert cfg.episode_length_s == standard.episode_length_s
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
    nerd = NerdFrankaLiftEnvCfg()
    assert cfg.scene.to_dict() == nerd.scene.to_dict()
    for field in ("observations", "actions", "events", "commands", "rewards", "terminations", "curriculum"):
        assert getattr(cfg, field).to_dict() == getattr(nerd, field).to_dict(), field


def test_franka_lift_scene_preserves_parent_robot_overrides() -> None:
    cfg = FrankaCubeLiftEnvCfg()
    stiffness = cfg.scene.robot.actuators["panda_shoulder"].stiffness
    assert isinstance(stiffness, (float, int))
    cfg.scene.robot.actuators["panda_shoulder"].stiffness = stiffness * 2
    expected = cfg.scene.robot.to_dict()
    apply_franka_lift_nerd_scene(cfg)
    assert cfg.scene.robot.to_dict() == expected
    assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions is True


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton MJWarp requires CUDA")
def test_franka_lift_resting_cube_does_not_earn_lifting_reward() -> None:
    from isaaclab_neural.generate.adapter import DataGenerationAdapter
    from isaaclab_neural.physics import NerdSolverCfg
    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    from isaaclab.app import launch_simulation
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

    from isaaclab_tasks.contrib.lift import mdp as lift_mdp

    cfg = resolve_presets(FrankaLiftCubeDatasetGenEnvCfg(), {"newton_mjwarp"})
    cfg.scene.num_envs = 2
    cfg.seed = 0
    cfg.commands.object_pose.debug_vis = False
    with launch_simulation(cfg, {"device": "cuda:0"}):
        with newton_material_binding_api_autofix():
            env: ManagerBasedRLEnv = cast(
                ManagerBasedRLEnv, gym.make("Isaac-Lift-Cube-Franka-Dataset-Gen-v0", cfg=cfg).unwrapped
            )
        try:
            assert isinstance(env, ManagerBasedRLEnv)
            env.reset()
            adapter = DataGenerationAdapter(
                env, NerdSolverCfg(name="NeuralSolver", states_frame="world", num_contacts_per_env=64)
            )
            assert adapter.solver.articulations_per_env == 2
            assert adapter.solver.dof_q_per_env == len(JOINT_Q_MIN["Franka-Lift"])
            assert adapter.solver.dof_qd_per_env == len(JOINT_QD_LIM["Franka-Lift"])
            actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
            actions[:, -1] = 1.0
            for _ in range(round(2.0 / env.step_dt)):
                obs, reward, terminated, truncated, _ = adapter.step(actions)
                assert torch.isfinite(obs["policy"]).all()
                assert torch.isfinite(reward).all()
                assert not (terminated | truncated).any()
            inputs = adapter.raw_neural_inputs()
            assert torch.isfinite(inputs["contact_depths"]).all()
            assert inputs["contact_normals"].abs().sum() > 0
            height = env.scene["object"].data.root_pos_w.torch[:, 2]
            surface = cfg.scene.table.init_state.pos[2] + cfg.scene.table.spawn.size[2] / 2
            assert torch.all(height > surface)
            lifted = lift_mdp.object_is_lifted(env, **cfg.rewards.lifting_object.params)
            assert not lifted.any(), height.tolist()
        finally:
            env.close()


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
