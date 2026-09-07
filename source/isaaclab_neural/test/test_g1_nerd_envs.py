# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Unitree G1 NeRD env registration and cfg wiring."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

G1_NERD_TASKS = (
    "Isaac-Velocity-Flat-G1-NeRD-v0",
    "Isaac-Velocity-Rough-G1-NeRD-v0",
)
G1_DATASET_GEN_TASKS = (
    "Isaac-Velocity-Flat-G1-Dataset-Gen-v0",
    "Isaac-Velocity-Rough-G1-Dataset-Gen-v0",
)


@pytest.fixture(scope="module", autouse=True)
def _import_nerd_envs():
    """Ensure NeRD env package registers gym ids."""
    import isaaclab_neural.envs  # noqa: F401


@pytest.mark.parametrize("task_id", G1_NERD_TASKS + G1_DATASET_GEN_TASKS)
def test_g1_tasks_are_registered(task_id: str):
    assert task_id in gym.registry


def test_g1_flat_nerd_cfg_uses_newton_native_by_default():
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg

    cfg = NerdG1FlatEnvCfg()
    assert cfg.sim.physics.solver_cfg.contact_mode == "newton_native"
    assert cfg.sim.physics.solver_cfg.num_contacts_per_env == 64
    assert cfg.scene.contact_forces is None
    assert cfg.rewards.feet_slide is None
    assert cfg.rewards.feet_air_time is not None
    assert cfg.terminations.base_contact is not None
    assert cfg.terminations.base_contact.params["body_cfg"].body_names == "torso_link"


def test_g1_rough_nerd_cfg_uses_newton_native_geometry_mdp():
    from isaaclab_neural.envs.g1_nerd_env import NerdG1RoughEnvCfg

    cfg = NerdG1RoughEnvCfg()
    assert cfg.sim.physics.solver_cfg.contact_mode == "newton_native"
    assert cfg.sim.physics.solver_cfg.num_contacts_per_env == 64
    assert cfg.scene.contact_forces is None
    assert cfg.rewards.feet_slide is None
    assert cfg.rewards.feet_air_time is not None
    assert cfg.terminations.base_contact is not None
    assert cfg.rewards.feet_air_time.params["body_cfg"].body_names == ".*_ankle_roll_link"
    assert cfg.terminations.base_contact.params["body_cfg"].body_names == "torso_link"


def test_g1_flat_runtime_solver_override_reconfigures_geometry_mdp():
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnv, NerdG1FlatEnvCfg
    from isaaclab_neural.physics import NerdSolverCfg

    cfg = NerdG1FlatEnvCfg()
    cfg.sim.physics.solver_cfg = NerdSolverCfg(contact_mode="newton_native", num_contacts_per_env=64)
    NerdG1FlatEnv._configure_solver_dependent_mdp(None, cfg)

    assert cfg.rewards.feet_air_time is not None
    assert cfg.terminations.base_contact is not None

    # ``NerdSolverCfg`` rejects deprecated ``fixed_ground``. Mutate the live
    # field to confirm non-native overrides still drop force-free MDP terms.
    cfg.sim.physics.solver_cfg.contact_mode = "unsupported"
    NerdG1FlatEnv._configure_solver_dependent_mdp(None, cfg)
    assert cfg.rewards.feet_air_time is None
    assert cfg.terminations.base_contact is None


def test_g1_token_runtime_base_contact_detects_torso(monkeypatch):
    from isaaclab_neural.contacts.contact_set_schema import (
        CONTACT_TOKEN_BODY_SLOT_INDEX,
        CONTACT_TOKEN_DIM,
        CONTACT_TOKEN_GAP_INDEX,
        CONTACT_TOKEN_VALID_INDEX,
    )
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnv, NerdG1FlatEnvCfg
    from isaaclab_neural.physics import NerdSolverCfg, NewtonNerdManager

    cfg = NerdG1FlatEnvCfg()
    cfg.sim.physics.solver_cfg = NerdSolverCfg(
        contact_mode="newton_native",
        contact_representation="contact_tokens",
        num_contacts_per_env=64,
    )
    NerdG1FlatEnv._configure_solver_dependent_mdp(None, cfg)

    tokens = torch.zeros((1, 2, CONTACT_TOKEN_DIM))
    tokens[0, 0, CONTACT_TOKEN_VALID_INDEX] = 1.0
    tokens[0, 0, CONTACT_TOKEN_BODY_SLOT_INDEX] = 2
    tokens[0, 0, CONTACT_TOKEN_GAP_INDEX] = -0.01
    adapter = SimpleNamespace(
        bodies_per_env=4,
        contact_representation="contact_tokens",
        contact_tokens=tokens,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(NewtonNerdManager, "_nerd_contact_adapter", adapter)

    term = cfg.terminations.base_contact
    assert term is not None
    term.params["body_cfg"].body_ids = [2]

    actual = term.func(SimpleNamespace(), **term.params)

    torch.testing.assert_close(actual, torch.tensor([True]))


def test_g1_token_runtime_base_contact_uses_newton_body_ids(monkeypatch):
    from isaaclab_neural.contacts.contact_set_schema import (
        CONTACT_TOKEN_BODY_SLOT_INDEX,
        CONTACT_TOKEN_DIM,
        CONTACT_TOKEN_GAP_INDEX,
        CONTACT_TOKEN_VALID_INDEX,
    )
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnv, NerdG1FlatEnvCfg
    from isaaclab_neural.physics import NerdSolverCfg, NewtonNerdManager

    cfg = NerdG1FlatEnvCfg()
    cfg.sim.physics.solver_cfg = NerdSolverCfg(
        contact_mode="newton_native",
        contact_representation="contact_tokens",
        num_contacts_per_env=64,
    )
    NerdG1FlatEnv._configure_solver_dependent_mdp(None, cfg)

    tokens = torch.zeros((1, 1, CONTACT_TOKEN_DIM))
    tokens[0, 0, CONTACT_TOKEN_VALID_INDEX] = 1.0
    tokens[0, 0, CONTACT_TOKEN_BODY_SLOT_INDEX] = 0
    tokens[0, 0, CONTACT_TOKEN_GAP_INDEX] = -0.01
    body_local_ids = torch.full((6,), -1, dtype=torch.long)
    body_local_ids[5] = 2
    adapter = SimpleNamespace(
        bodies_per_env=4,
        contact_representation="contact_tokens",
        contact_tokens=tokens,
        contact_token_body_ids=torch.tensor([[5]]),
        body_local_ids=body_local_ids,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(NewtonNerdManager, "_nerd_contact_adapter", adapter)

    term = cfg.terminations.base_contact
    assert term is not None
    term.params["body_cfg"].body_ids = [2]

    actual = term.func(SimpleNamespace(), **term.params)

    torch.testing.assert_close(actual, torch.tensor([True]))


def test_g1_dataset_gen_disables_force_sensor_mdp():
    from isaaclab_neural.envs.g1_dataset_gen_cfg import G1DatasetGenFlatEnvCfg, G1DatasetGenRoughEnvCfg

    import isaaclab.envs.mdp as base_mdp

    from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    flat = G1DatasetGenFlatEnvCfg()
    rough = G1DatasetGenRoughEnvCfg()
    for cfg in (flat, rough):
        assert cfg.rewards.feet_air_time is None
        assert cfg.rewards.feet_slide is None
        assert cfg.terminations.base_contact is None
        assert cfg.terminations.low_root_height is not None
        assert cfg.events.reset_robot_joints.func is base_mdp.reset_joints_by_offset
        assert cfg.events.reset_robot_joints.params["position_range"] == (-0.2, 0.2)
        assert cfg.events.reset_robot_joints.params["velocity_range"] == (-0.1, 0.1)
        assert cfg.events.reset_base.params["pose_range"]["z"] == (-0.1, 0.1)

    rl_cfg = G1FlatEnvCfg()
    assert rl_cfg.events.reset_robot_joints.func is base_mdp.reset_joints_by_scale
    assert rl_cfg.events.reset_robot_joints.params["position_range"] == (1.0, 1.0)
    assert "z" not in rl_cfg.events.reset_base.params["pose_range"]


def test_g1_nerd_and_dataset_gen_use_stock_robot():
    from isaaclab_assets import G1_MINIMAL_CFG

    from isaaclab_neural.envs.g1_dataset_gen_cfg import G1DatasetGenFlatEnvCfg, G1DatasetGenRoughEnvCfg
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg, NerdG1RoughEnvCfg

    for cfg in (
        NerdG1FlatEnvCfg(),
        NerdG1RoughEnvCfg(),
        G1DatasetGenFlatEnvCfg(),
        G1DatasetGenRoughEnvCfg(),
    ):
        assert cfg.scene.robot.spawn.usd_path == G1_MINIMAL_CFG.spawn.usd_path
        assert cfg.rewards.joint_deviation_fingers is not None
        arm_joints = cfg.scene.robot.actuators["arms"].joint_names_expr
        assert ".*_five_joint" in arm_joints
        assert ".*_elbow_pitch_joint" in arm_joints


def test_g1_dataset_gen_uses_rich_velocity_commands():
    from isaaclab_neural.envs.g1_dataset_gen_cfg import G1DatasetGenFlatEnvCfg, G1DatasetGenRoughEnvCfg
    from isaaclab_neural.robots.g1 import (
        G1_DATASET_COMMAND_ANG_VEL_Z,
        G1_DATASET_COMMAND_LIN_VEL_X,
        G1_DATASET_COMMAND_LIN_VEL_Y,
        G1_DATASET_COMMAND_RESAMPLE_S,
    )

    for cfg in (G1DatasetGenFlatEnvCfg(), G1DatasetGenRoughEnvCfg()):
        assert cfg.commands.base_velocity.ranges.lin_vel_x == G1_DATASET_COMMAND_LIN_VEL_X
        assert cfg.commands.base_velocity.ranges.lin_vel_y == G1_DATASET_COMMAND_LIN_VEL_Y
        assert cfg.commands.base_velocity.ranges.ang_vel_z == G1_DATASET_COMMAND_ANG_VEL_Z
        assert cfg.commands.base_velocity.ranges.heading is None
        assert cfg.commands.base_velocity.heading_command is False
        assert cfg.commands.base_velocity.resampling_time_range == G1_DATASET_COMMAND_RESAMPLE_S
        assert cfg.commands.base_velocity.rel_standing_envs == 0.0


def test_g1_nerd_keeps_stock_flat_velocity_commands():
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg, NerdG1RoughEnvCfg
    from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
    from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

    stock_flat = G1FlatEnvCfg()
    nerd_flat = NerdG1FlatEnvCfg()
    assert nerd_flat.commands.base_velocity.ranges.lin_vel_x == stock_flat.commands.base_velocity.ranges.lin_vel_x
    assert nerd_flat.commands.base_velocity.heading_command is True
    assert nerd_flat.commands.base_velocity.resampling_time_range == (10.0, 10.0)

    stock_rough = G1RoughEnvCfg()
    nerd_rough = NerdG1RoughEnvCfg()
    assert nerd_rough.commands.base_velocity.ranges.lin_vel_y == stock_rough.commands.base_velocity.ranges.lin_vel_y
    assert nerd_rough.commands.base_velocity.heading_command is True


def test_g1_rough_nerd_task_uses_rough_agent_cfg():
    from isaaclab_neural.envs.agents.g1_nerd_rsl_rl_ppo_cfg import G1RoughNeRDPPORunnerCfg

    task_spec = gym.spec("Isaac-Velocity-Rough-G1-NeRD-v0")
    agent_cfg = G1RoughNeRDPPORunnerCfg()

    assert task_spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":G1RoughNeRDPPORunnerCfg")
    assert agent_cfg.experiment_name == "g1_rough_nerd"
