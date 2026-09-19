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
    FrankaReachPolicyEnvCfg,
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

from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import FRANKA_PANDA_LIFT_CFG, FrankaLiftEnvCfg
from isaaclab_tasks.core.lift.lift_env_cfg import PhysicsCfg as LiftPhysicsCfg
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


@pytest.mark.parametrize("cfg_type", [FrankaReachDatasetGenEnvCfg, FrankaReachPolicyEnvCfg, NerdFrankaReachEnvCfg])
def test_franka_reach_pipeline_uses_core_lift_robot_at_60_hz(cfg_type) -> None:
    cfg = resolve_presets(cfg_type(), {"newton_mjwarp"})
    assert cfg.scene.robot.spawn.usd_path == FRANKA_PANDA_LIFT_CFG.spawn.usd_path
    assert cfg.scene.robot.spawn.variants == FRANKA_PANDA_LIFT_CFG.spawn.variants
    assert set(cfg.scene.robot.actuators) == {"panda_arm", "panda_hand", "panda_finger2_passive"}
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
    expected_robot = FRANKA_PANDA_LIFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    expected_robot.spawn.articulation_props.enabled_self_collisions = False
    expected_robot.soft_joint_pos_limit_factor = FRANKA_REACH_DATASET_SOFT_JOINT_POS_LIMIT_FACTOR
    assert cfg.scene.robot.to_dict() == expected_robot.to_dict()
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


def test_franka_lift_nerd_uses_core_task_with_nominal_physics() -> None:
    cfg = NerdFrankaLiftEnvCfg()
    standard = FrankaLiftEnvCfg()

    assert cfg.scene.robot.spawn.usd_path == FRANKA_PANDA_LIFT_CFG.spawn.usd_path
    clearance = cfg.events.conditional_reset.params["valid_criteria"]["robot_table_clearance"]
    assert clearance.body_names == ["panda_link[2-7]", "panda_hand", ".*finger"]
    assert cfg.scene.table.spawn.size == (0.8, 1.5, 0.5)
    assert cfg.scene.table.init_state.pos[2] + cfg.scene.table.spawn.size[2] / 2 == pytest.approx(0.255)
    assert cfg.scene.table.spawn.rigid_props is None
    assert cfg.scene.object.spawn.size == (0.05, 0.05, 0.05)
    for field in ("observations", "actions", "commands", "rewards"):
        assert getattr(cfg, field).to_dict() == getattr(standard, field).to_dict(), field
    assert cfg.curriculum is None
    assert cfg.events.variable_gravity is None
    assert cfg.events.joint_stiffness_and_damping is None
    assert cfg.terminations.object_out_of_bound.params["in_bound_range"]["z"] == (0.0, 2.0)
    assert cfg.sim.dt == standard.sim.dt == 1.0 / 120.0
    assert cfg.decimation == standard.decimation == 4
    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    assert cfg.sim.physics.solver_cfg.num_contacts_per_env == 64
    assert cfg.sim.physics.solver_cfg.states_frame == "body"
    assert cfg.sim.physics.solver_cfg.contact_representation == "contact_tokens"
    preset_names = {name for fields in collect_presets(cfg).values() for name in fields}
    assert "newton_mjwarp" in preset_names


def test_franka_lift_dataset_gen_exposes_core_newton_mjwarp() -> None:
    cfg = FrankaLiftCubeDatasetGenEnvCfg()

    assert cfg.scene.table.spawn.size == (0.8, 1.5, 0.5)
    assert cfg.scene.table.init_state.pos[2] + cfg.scene.table.spawn.size[2] / 2 == pytest.approx(0.255)
    assert isinstance(cfg.sim.physics, LiftPhysicsCfg)
    assert isinstance(cfg.sim.physics.newton_mjwarp, NewtonCfg)
    assert cfg.sim.physics.newton_mjwarp.solver_cfg.nconmax == 200
    assert cfg.sim.physics.newton_mjwarp.use_cuda_graph is False
    assert cfg.actions.action.scale == 0.1
    assert gym.spec("Isaac-Lift-Cube-Franka-Dataset-Gen-v0").kwargs["default_agent"] == "rsl_rl"


def test_franka_lift_scene_preserves_core_robot_drives() -> None:
    cfg = FrankaLiftEnvCfg()
    expected = cfg.scene.robot.actuators["panda_arm"].to_dict()
    apply_franka_lift_nerd_scene(cfg)
    assert cfg.scene.robot.actuators["panda_arm"].to_dict() == expected
    assert cfg.scene.robot.spawn.variants == FRANKA_PANDA_LIFT_CFG.spawn.variants


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton MJWarp requires CUDA")
def test_franka_lift_dataset_gen_has_joint_robot_free_cube_and_static_table() -> None:
    from isaaclab_neural.generate.adapter import DataGenerationAdapter
    from isaaclab_neural.physics import NerdSolverCfg
    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    from isaaclab.app import launch_simulation
    from isaaclab.envs import ManagerBasedRLEnv

    cfg = resolve_presets(FrankaLiftCubeDatasetGenEnvCfg(), {"newton_mjwarp"})
    cfg.scene.num_envs = 2
    cfg.seed = 0
    with launch_simulation(cfg, {"device": "cuda:0", "visualizer": []}):
        with newton_material_binding_api_autofix():
            env = cast(ManagerBasedRLEnv, gym.make("Isaac-Lift-Cube-Franka-Dataset-Gen-v0", cfg=cfg).unwrapped)
            env.reset()
        try:
            adapter = DataGenerationAdapter(
                env,
                NerdSolverCfg(
                    name="NeuralSolver",
                    states_frame="body",
                    contact_mode="newton_native",
                    contact_representation="contact_tokens",
                    num_contacts_per_env=64,
                    max_contact_tokens=64,
                    contact_packing_policy="body_round_robin_pair_atomic",
                ),
            )
            assert adapter.solver.articulations_per_env == 2
            assert (adapter.state_dim, adapter.joint_f_dim) == (31, 15)
            assert adapter.solver.free_joint_indices.tolist() == [11]
            assert adapter.contact_adapter is not None
            assert adapter.contact_adapter.num_primary_bodies_per_env == 12
            action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
            for frame in range(cfg.decimation):
                adapter.step_action_frame(action, process_action=frame == 0)
            inputs = adapter.raw_neural_inputs()
            assert torch.isfinite(inputs["states"]).all()
            assert torch.isfinite(inputs["contact_tokens"]).all()
        finally:
            env.close()


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton MJWarp requires CUDA")
def test_lift_predicted_normal_forces_feed_policy_rewards_and_partial_reset() -> None:
    from isaaclab_neural.contacts.finger_object_force import SENSOR_NAMES, TARGET_KEY, read_normal_forces
    from isaaclab_neural.models.models import ModelMixedInputOutput
    from isaaclab_neural.physics.nerd_newton_manager import NewtonNerdManager
    from isaaclab_neural.train.train import build_launch_cfg
    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.core.lift.mdp.observations import fingers_contact_force_b

    cfg = resolve_presets(NerdFrankaLiftEnvCfg(), {"newton_mjwarp"})
    cfg.scene.num_envs = 2
    cfg.seed = 0
    cfg.sim.physics.solver_cfg.name = "NeuralSolver"
    cfg.events.conditional_reset.params["buffer_size_per_group"] = 4
    cfg.events.conditional_reset.params["max_prefill_iters"] = 100
    cfg.observations.perception.object_point_cloud.params["visualize"] = False
    with launch_simulation(build_launch_cfg(cfg), {"device": "cuda:0", "visualizer": []}):
        with NewtonNerdManager._defer_neural_model_loading(), newton_material_binding_api_autofix():
            env = gym.make("Isaac-Lift-Cube-Franka-NeRD-v0", cfg=cfg).unwrapped
        try:
            solver = env.neural_adapter.solver
            force = torch.tensor(
                [[[30.0, 0.0, 0.0], [0.0, -4.0, 0.0]], [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]], device=env.device
            )

            class PrescribedForce(torch.nn.Module):
                predict_finger_object_normal_forces = True

                def forward_with_aux(self, inputs, *, single_step):
                    current = inputs["states"][:, -1:]
                    return ModelMixedInputOutput(
                        solver.convert_next_states_to_prediction(current, current), {TARGET_KEY: force[:, None]}
                    )

            solver.set_neural_solver_model(PrescribedForce())
            env.reset()
            action = torch.zeros(2, 9, device=env.device)
            for _ in range(6):
                observations, *_ = env.step(action)
            torch.testing.assert_close(fingers_contact_force_b(env, list(SENSOR_NAMES)), force.flatten(1))
            torch.testing.assert_close(
                observations["proprio"][:, -30:].reshape(2, 5, 6),
                force.clamp(-20.0, 20.0).flatten(1)[:, None].expand(-1, 5, -1),
            )
            term = cfg.rewards.good_finger_contact
            torch.testing.assert_close(term.func(env, **term.params), torch.tensor([True, False], device=env.device))
            before = read_normal_forces(env.scene).clone()
            env.reset(env_ids=[0])
            after = read_normal_forces(env.scene)
            torch.testing.assert_close(after[0], torch.zeros_like(after[0]))
            torch.testing.assert_close(after[1], before[1])
            torch.testing.assert_close(env.obs_buf["proprio"][0, -30:], torch.zeros(30, device=env.device))
            solver.set_neural_solver_model(torch.nn.Identity())
            with pytest.raises(RuntimeError, match="filtered finger-object"):
                NewtonNerdManager.publish_finger_forces()
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
        "Isaac-Reach-Franka-Policy-v0",
        "Isaac-Lift-Cube-Franka-NeRD-v0",
        "Isaac-Lift-Cube-Franka-Dataset-Gen-v0",
    ):
        assert env_id in gym.registry
