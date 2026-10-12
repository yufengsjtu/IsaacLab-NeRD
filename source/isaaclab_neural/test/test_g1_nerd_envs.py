# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Unitree G1 NeRD env registration and cfg wiring."""

from __future__ import annotations

import math
import os
from argparse import Namespace
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from isaaclab_neural.physics import NerdNewtonCfg

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
    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    assert cfg.sim.physics.solver_cfg.contact_mode == "newton_native"
    assert cfg.sim.physics.solver_cfg.num_contacts_per_env == 64
    assert cfg.scene.contact_forces is None
    assert cfg.rewards.feet_slide.weight == -0.1
    assert cfg.rewards.feet_air_time is not None
    assert cfg.terminations.base_contact is not None
    assert cfg.terminations.base_contact.params["body_cfg"].body_names == "torso_link"


def test_g1_rough_nerd_cfg_uses_newton_native_geometry_mdp():
    from isaaclab_neural.envs.g1_nerd_env import NerdG1RoughEnvCfg

    cfg = NerdG1RoughEnvCfg()
    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    assert cfg.sim.physics.solver_cfg.contact_mode == "newton_native"
    assert cfg.sim.physics.solver_cfg.num_contacts_per_env == 64
    assert cfg.scene.contact_forces is None
    assert cfg.rewards.feet_slide.weight == 0.0
    assert cfg.rewards.feet_air_time is not None
    assert cfg.terminations.base_contact is not None
    assert cfg.rewards.feet_air_time.params["body_cfg"].body_names == ".*_ankle_roll_link"
    assert cfg.terminations.base_contact.params["body_cfg"].body_names == "torso_link"


@pytest.mark.parametrize("terrain", ("flat", "rough"))
def test_g1_nerd_inherits_stock_collision_configuration(terrain: str):
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg, NerdG1RoughEnvCfg
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab.utils.dict import class_to_dict

    from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
    from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    stock_cfg = G1FlatEnvCfg() if terrain == "flat" else G1RoughEnvCfg()
    nerd_cfg = NerdG1FlatEnvCfg() if terrain == "flat" else NerdG1RoughEnvCfg()
    stock_physics = resolve_presets(stock_cfg.sim.physics)
    nerd_physics = nerd_cfg.sim.physics
    assert stock_physics is not None
    assert isinstance(stock_physics, NewtonCfg)
    assert isinstance(nerd_physics, NerdNewtonCfg)

    assert class_to_dict(nerd_physics.default_shape_cfg) == class_to_dict(stock_physics.default_shape_cfg)
    assert class_to_dict(nerd_physics.collision_cfg) == class_to_dict(stock_physics.collision_cfg)
    assert nerd_physics.default_shape_cfg is not stock_physics.default_shape_cfg
    assert nerd_physics.collision_cfg is not stock_physics.collision_cfg
    assert nerd_physics.num_substeps == 1
    assert nerd_physics.use_cuda_graph is False
    assert nerd_physics.solver_cfg.name == "TransformerNeuralSolver"


@pytest.mark.parametrize("task_id", G1_NERD_TASKS)
@pytest.mark.parametrize("mdp_override", ("default", "custom", "disabled"))
@pytest.mark.parametrize("restore_checkpoint", (False, True))
def test_g1_rl_solver_overrides_preserve_task_mdp(
    task_id: str, mdp_override: str, restore_checkpoint: bool, monkeypatch
):
    from isaaclab_neural.envs import neural_env_wrapper
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnv, NerdG1FlatEnvCfg, NerdG1RoughEnv, NerdG1RoughEnvCfg
    from isaaclab_neural.rl.rsl_rl import cli_args

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.dict import class_to_dict

    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    assert isinstance(cfg, (NerdG1FlatEnvCfg, NerdG1RoughEnvCfg))
    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    if mdp_override == "custom":
        cfg.rewards.feet_air_time.weight = 0.0
        cfg.rewards.feet_air_time.params["threshold"] = 0.9
        cfg.rewards.feet_air_time.params["body_cfg"].body_names = "left_ankle_roll_link"
        cfg.rewards.feet_slide.weight = 0.0
        cfg.terminations.base_contact.params["body_cfg"].body_names = "pelvis"
    elif mdp_override == "disabled":
        setattr(cfg.rewards, "feet_air_time", None)
        setattr(cfg.rewards, "feet_slide", None)
        setattr(cfg.terminations, "base_contact", None)
    expected_rewards = class_to_dict(cfg.rewards)
    expected_terminations = class_to_dict(cfg.terminations)
    checkpoint = {"cfg": {"env": {"neural_solver_cfg": class_to_dict(cfg.sim.physics.solver_cfg)}}}
    monkeypatch.setattr(cli_args, "load_checkpoint", lambda *args, **kwargs: checkpoint)
    args = Namespace(
        neural_model_path="dynamics.pt" if restore_checkpoint else None,
        contact_mode="newton_native",
        num_contacts_per_env=32,
        use_kv_cache=False,
    )

    solver_cfg = cli_args.apply_neural_model_to_env_cfg(cfg, args)
    initialized = []

    def initialize_env(self, cfg, **kwargs):
        initialized.append(cfg)
        assert class_to_dict(cfg.rewards) == expected_rewards
        assert class_to_dict(cfg.terminations) == expected_terminations

    monkeypatch.setattr(ManagerBasedRLEnv, "__init__", initialize_env)
    monkeypatch.setattr(neural_env_wrapper, "NeuralEnvAdapter", lambda env: None)
    env_cls = NerdG1FlatEnv if isinstance(cfg, NerdG1FlatEnvCfg) else NerdG1RoughEnv
    env_cls(cfg, solver_cfg=solver_cfg)

    assert initialized == [cfg]
    assert cfg.sim.physics.solver_cfg.num_contacts_per_env == 32
    if restore_checkpoint:
        assert solver_cfg is not None
        assert cfg.sim.physics.solver_cfg is solver_cfg
        assert solver_cfg.neural_model_path == args.neural_model_path
    assert class_to_dict(cfg.rewards) == expected_rewards
    assert class_to_dict(cfg.terminations) == expected_terminations


@pytest.mark.parametrize("task_id", G1_NERD_TASKS)
def test_g1_runtime_rejects_invalid_contact_mode_before_simulation(task_id: str, monkeypatch):
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnv, NerdG1FlatEnvCfg, NerdG1RoughEnv, NerdG1RoughEnvCfg

    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    assert isinstance(cfg, (NerdG1FlatEnvCfg, NerdG1RoughEnvCfg))
    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    setattr(cfg.sim.physics.solver_cfg, "contact_mode", "unsupported")
    monkeypatch.setattr(
        ManagerBasedRLEnv,
        "__init__",
        lambda *args, **kwargs: pytest.fail("Invalid solver reached simulation initialization"),
    )
    env_cls = NerdG1FlatEnv if isinstance(cfg, NerdG1FlatEnvCfg) else NerdG1RoughEnv

    with pytest.raises(ValueError, match="Unsupported contact mode"):
        env_cls(cfg)


def test_g1_contact_mode_helper_reconfigures_geometry_mdp():
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg, apply_g1_contact_mode_mdp
    from isaaclab_neural.physics import NerdSolverCfg

    cfg = NerdG1FlatEnvCfg()
    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    cfg.sim.physics.solver_cfg = NerdSolverCfg(contact_mode="newton_native", num_contacts_per_env=64)
    apply_g1_contact_mode_mdp(cfg)

    assert cfg.rewards.feet_air_time is not None
    assert cfg.terminations.base_contact is not None

    # ``NerdSolverCfg`` rejects deprecated ``fixed_ground``. Mutate the live
    # field to confirm non-native overrides still drop force-free MDP terms.
    setattr(cfg.sim.physics.solver_cfg, "contact_mode", "unsupported")
    apply_g1_contact_mode_mdp(cfg)
    assert cfg.rewards.feet_air_time is None
    assert cfg.terminations.base_contact is None
    setattr(cfg.sim.physics.solver_cfg, "contact_mode", "newton_native")
    apply_g1_contact_mode_mdp(cfg)
    assert cfg.rewards.feet_air_time.weight == 0.75
    assert cfg.rewards.feet_slide.weight == -0.1
    assert cfg.terminations.base_contact is not None


def test_g1_token_runtime_base_contact_detects_torso(monkeypatch):
    from isaaclab_neural.contacts.contact_set_schema import (
        CONTACT_TOKEN_BODY_SLOT_INDEX,
        CONTACT_TOKEN_DIM,
        CONTACT_TOKEN_GAP_INDEX,
        CONTACT_TOKEN_VALID_INDEX,
    )
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg, apply_g1_contact_mode_mdp
    from isaaclab_neural.physics import NerdSolverCfg, NewtonNerdManager

    cfg = NerdG1FlatEnvCfg()
    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    cfg.sim.physics.solver_cfg = NerdSolverCfg(
        contact_mode="newton_native",
        contact_representation="contact_tokens",
        num_contacts_per_env=64,
    )
    apply_g1_contact_mode_mdp(cfg)

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
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg, apply_g1_contact_mode_mdp
    from isaaclab_neural.physics import NerdSolverCfg, NewtonNerdManager

    cfg = NerdG1FlatEnvCfg()
    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    cfg.sim.physics.solver_cfg = NerdSolverCfg(
        contact_mode="newton_native",
        contact_representation="contact_tokens",
        num_contacts_per_env=64,
    )
    apply_g1_contact_mode_mdp(cfg)

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
        assert getattr(cfg.terminations, "low_root_height") is not None
        assert cfg.events.reset_robot_joints.func is base_mdp.reset_joints_by_offset
        assert cfg.events.reset_robot_joints.params["position_range"] == (-0.2, 0.2)
        assert cfg.events.reset_robot_joints.params["velocity_range"] == (-0.1, 0.1)
        pose_range = cfg.events.reset_base.params["pose_range"]
        assert isinstance(pose_range, dict)
        assert pose_range["z"] == (-0.1, 0.1)

    rl_cfg = G1FlatEnvCfg()
    assert rl_cfg.events.reset_robot_joints.func is base_mdp.reset_joints_by_scale
    assert rl_cfg.events.reset_robot_joints.params["position_range"] == (1.0, 1.0)
    pose_range = rl_cfg.events.reset_base.params["pose_range"]
    assert isinstance(pose_range, dict)
    assert "z" not in pose_range


def test_g1_nerd_and_dataset_gen_use_stock_robot():
    from isaaclab_neural.envs.g1_dataset_gen_cfg import G1DatasetGenFlatEnvCfg, G1DatasetGenRoughEnvCfg
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg, NerdG1RoughEnvCfg

    from isaaclab.sim import UsdFileCfg

    from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG

    for cfg in (
        NerdG1FlatEnvCfg(),
        NerdG1RoughEnvCfg(),
        G1DatasetGenFlatEnvCfg(),
        G1DatasetGenRoughEnvCfg(),
    ):
        assert isinstance(cfg.scene.robot.spawn, UsdFileCfg)
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


@pytest.mark.parametrize("terrain", ("flat", "rough"))
def test_g1_biped_reward_keeps_stock_weight_and_threshold(terrain):
    from isaaclab_neural.contacts.geometry_mdp import GeometryFeetAirTimePositiveBiped, GeometryFeetSlide
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg, NerdG1RoughEnvCfg

    from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
    from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

    stock = G1FlatEnvCfg() if terrain == "flat" else G1RoughEnvCfg()
    cfg = NerdG1FlatEnvCfg() if terrain == "flat" else NerdG1RoughEnvCfg()
    assert cfg.rewards.feet_air_time.func is GeometryFeetAirTimePositiveBiped
    assert cfg.rewards.feet_air_time.weight == stock.rewards.feet_air_time.weight
    assert cfg.rewards.feet_air_time.params["threshold"] == stock.rewards.feet_air_time.params["threshold"]
    assert cfg.rewards.feet_slide.func is GeometryFeetSlide
    expected_slide_weight = stock.rewards.feet_slide.weight if terrain == "flat" else 0.0
    assert cfg.rewards.feet_slide.weight == expected_slide_weight
    assert cfg.rewards.feet_slide.params["history_length"] == stock.scene.contact_forces.history_length
    if terrain == "flat":
        assert {name: term.weight for name, term in vars(cfg.rewards).items() if term is not None} == {
            name: term.weight for name, term in vars(stock.rewards).items() if term is not None
        }


def test_g1_flat_invalid_state_terminates_finite_outliers_and_rejects_nan():
    from isaaclab_neural.envs.g1_nerd_env import NerdG1FlatEnvCfg

    cfg = NerdG1FlatEnvCfg()
    term = cfg.terminations.invalid_state
    assert not term.time_out
    root = torch.zeros((5, 13))
    root[:, 2] = 0.7
    root[:, 6] = 1.0
    joints = torch.zeros((5, 37))
    velocities = torch.zeros_like(joints)
    data = SimpleNamespace(
        root_state_w=SimpleNamespace(torch=root),
        joint_pos=SimpleNamespace(torch=joints),
        joint_vel=SimpleNamespace(torch=velocities),
    )
    scene = {"robot": SimpleNamespace(data=data)}

    class Scene(dict):
        env_origins = torch.zeros((5, 3))

    env = SimpleNamespace(scene=Scene(scene))
    root[1, 2] = 2.0
    root[2, 7] = 21.0
    root[3, 10] = 51.0
    velocities[4, 0] = 101.0
    torch.testing.assert_close(term.func(env, **term.params), torch.tensor([False, True, True, True, True]))
    root[0, 2] = float("nan")
    with pytest.raises(FloatingPointError, match="Non-finite G1 state"):
        term.func(env, **term.params)


def test_rough_mbpo_scene_is_fixed_and_does_not_reuse_flat_height_guard():
    from isaaclab_neural.envs.g1_nerd_env import NerdG1RoughEnvCfg, configure_g1_mbpo_rough

    cfg = NerdG1RoughEnvCfg()
    original = NerdG1RoughEnvCfg()
    configure_g1_mbpo_rough(
        cfg,
        {
            "terrain_seed": 123,
            "terrain_rows": 2,
            "terrain_cols": 3,
            "terrain_difficulty_max": 0.25,
            "policy_observation_corruption": False,
        },
    )
    generator = cfg.scene.terrain.terrain_generator
    assert generator.seed == 123 and (generator.num_rows, generator.num_cols) == (2, 3)
    assert generator.difficulty_range == (0.0, 0.25) and not generator.curriculum
    assert cfg.curriculum.terrain_levels is None
    assert cfg.scene.height_scanner is not None and cfg.observations.policy.height_scan is not None
    assert not cfg.observations.policy.enable_corruption
    assert cfg.terminations.invalid_state.params["maximum_height"] is None
    assert original.scene.terrain.terrain_generator.num_rows == 10
    with pytest.raises(ValueError):
        configure_g1_mbpo_rough(cfg, {"terrain_difficulty_max": 1.1})


def test_aligned_rough_mbpo_keeps_stock_terrain_events_and_reward_weights():
    from isaaclab_neural.envs.g1_nerd_env import NerdG1RoughEnvCfg, configure_g1_mbpo_rough

    from isaaclab.utils.dict import class_to_dict

    from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

    stock, cfg = G1RoughEnvCfg(), NerdG1RoughEnvCfg()
    configure_g1_mbpo_rough(
        cfg, {"terrain_profile": "stock", "physics_events": "stock", "terrain_seed": 42, "role": "mj_collect"}
    )
    expected = class_to_dict(stock.scene.terrain.terrain_generator)
    actual = class_to_dict(cfg.scene.terrain.terrain_generator)
    expected["seed"] = actual["seed"] = 42
    assert actual == expected
    assert cfg.scene.terrain.max_init_terrain_level == stock.scene.terrain.max_init_terrain_level
    assert class_to_dict(cfg.events) == class_to_dict(stock.events)
    assert class_to_dict(cfg.curriculum) == class_to_dict(stock.curriculum)
    assert cfg.rewards.feet_slide.weight == stock.rewards.feet_slide.weight == -0.1
    assert cfg.mbpo_randomized_physics
    assert cfg.rewards.feet_air_time.params["maximum_separation"] == 0.0


def test_rough_mbpo_feet_contact_margin_applies_only_to_geometry_gait_rewards():
    from isaaclab_neural.envs.g1_nerd_env import NerdG1RoughEnvCfg, configure_g1_mbpo_rough

    from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

    cfg = NerdG1RoughEnvCfg()
    configure_g1_mbpo_rough(cfg, {"terrain_profile": "stock", "role": "nerd_train", "feet_contact_separation": 0.0025})
    for name in ("feet_air_time", "feet_slide"):
        assert getattr(cfg.rewards, name).params["maximum_separation"] == 0.0025
    assert "maximum_separation" not in cfg.terminations.base_contact.params
    stock = G1RoughEnvCfg()
    configure_g1_mbpo_rough(
        stock,
        {"terrain_profile": "stock", "role": "evaluate", "contact_mdp": "stock", "feet_contact_separation": 0.0025},
    )
    assert "maximum_separation" not in stock.rewards.feet_air_time.params
    with pytest.raises(ValueError, match="feet_contact_separation"):
        configure_g1_mbpo_rough(NerdG1RoughEnvCfg(), {"feet_contact_separation": 0.05})


def test_aligned_rough_runner_is_default_stock_runner():
    from isaaclab_neural.envs.agents.g1_nerd_rsl_rl_ppo_cfg import g1_mbpo_runner_cfg

    from isaaclab.utils.dict import class_to_dict

    from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1RoughPPORunnerCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    runner = g1_mbpo_runner_cfg("stock_rough")
    stock = resolve_presets(G1RoughPPORunnerCfg())
    assert class_to_dict(runner) == class_to_dict(stock)
    assert runner.max_iterations == 3000
    assert runner.actor.hidden_dims == runner.critic.hidden_dims == [512, 256, 128]


def test_g1_rough_nerd_task_uses_rough_agent_cfg():
    from isaaclab_neural.envs.agents.g1_nerd_rsl_rl_ppo_cfg import G1RoughNeRDPPORunnerCfg

    task_spec = gym.spec("Isaac-Velocity-Rough-G1-NeRD-v0")
    agent_cfg = G1RoughNeRDPPORunnerCfg()

    assert task_spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":G1RoughNeRDPPORunnerCfg")
    assert agent_cfg.experiment_name == "g1_rough_nerd"


G1_29DOF_BFM_TASKS = ("Isaac-G1-29DoF-BFM-Dataset-Gen-v0", "Isaac-G1-29DoF-BFM-Perturbed-Dataset-Gen-v0")


@pytest.mark.parametrize("task_id", G1_29DOF_BFM_TASKS)
def test_g1_29dof_bfm_tasks_are_registered(task_id: str):
    from isaaclab_neural.controllers.bfm_zero import BfmZeroLatentActionCfg

    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    assert isinstance(cfg.actions.bfm, BfmZeroLatentActionCfg)
    # BFM-Zero runs at 50 Hz on 200 Hz physics
    assert cfg.sim.dt * cfg.decimation == pytest.approx(0.02)
    assert cfg.actions.bfm.control_decimation == cfg.decimation
    assert cfg.terminations.invalid_state.params["maximum_height"] is None


def test_g1_29dof_perturbed_variant_adds_collection_perturbations():
    from isaaclab_neural.envs.g1_29dof_env_cfg import G129DofBfmEnvCfg, G129DofBfmPerturbedEnvCfg

    base, perturbed = G129DofBfmEnvCfg(), G129DofBfmPerturbedEnvCfg()
    assert base.events.root_pushes is None and base.events.actuator_gains is None
    assert base.events.reset_lying.params["fraction"] == 0.0
    assert base.actions.bfm.action_noise_std == 0.0
    step_dt = perturbed.sim.dt * perturbed.decimation
    assert perturbed.events.root_pushes.mode == "interval"
    assert perturbed.events.root_pushes.interval_range_s == pytest.approx((step_dt, step_dt))
    assert perturbed.events.actuator_gains.params["stiffness_distribution_params"] == (0.8, 1.2)
    assert perturbed.events.reset_lying.params["fraction"] == 0.3
    assert perturbed.actions.bfm.action_noise_std > 0.0


@pytest.mark.parametrize(
    "task_id",
    (
        "Isaac-Velocity-Flat-G1-29DoF-v0",
        "Isaac-Velocity-Flat-G1-29DoF-Dataset-Gen-v0",
        "Isaac-Velocity-Flat-G1-29DoF-NeRD-v0",
        "Isaac-G1-29DoF-Posture-v0",
    ),
)
def test_g1_29dof_velocity_tasks_swap_robot_and_joint_terms(task_id: str):
    from isaaclab_neural.robots.g1_29dof import G1_29DOF_USD_URL

    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    assert cfg.scene.robot.spawn.usd_path == G1_29DOF_USD_URL
    assert cfg.rewards.joint_deviation_fingers is None
    assert cfg.events.wrist_mass.params["operation"] == "abs"
    assert ".*_wrist_.*_joint" in cfg.rewards.joint_deviation_arms.params["asset_cfg"].joint_names


def test_g1_29dof_dataset_gen_keeps_the_teacher_command_and_reset_distribution():
    from isaaclab_neural.envs.g1_29dof_env_cfg import G129DofDatasetGenFlatEnvCfg

    from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    cfg, stock = G129DofDatasetGenFlatEnvCfg(), G1FlatEnvCfg()
    assert cfg.commands.base_velocity.ranges.lin_vel_x == stock.commands.base_velocity.ranges.lin_vel_x
    assert cfg.events.reset_robot_joints.params == stock.events.reset_robot_joints.params
    assert cfg.events.push_robot is None and cfg.terminations.base_contact is None


def test_g1_29dof_posture_task_commands_height_and_torso():
    from isaaclab_neural.envs.g1_29dof_env_cfg import G129DofPostureEnvCfg

    cfg = G129DofPostureEnvCfg()
    assert cfg.commands.base_velocity.rel_standing_envs == 1.0
    assert cfg.commands.posture.ranges.height[0] < cfg.commands.posture.ranges.height[1]
    assert cfg.observations.policy.posture_command.params["command_name"] == "posture"
    assert cfg.rewards.track_height.weight > 0.0 and cfg.rewards.track_torso.weight > 0.0
    assert cfg.rewards.feet_air_time is None


def test_g1_29dof_actuator_parameters_cover_every_joint_once():
    from isaaclab_neural.robots import g1_29dof

    from isaaclab.utils.string import resolve_matching_names_values

    actuator = g1_29dof.G1_29DOF_BFM_CFG.actuators["body"]
    for table in (actuator.stiffness, actuator.damping, actuator.armature, actuator.joint_effort_limit):
        ids, _, _ = resolve_matching_names_values(table, g1_29dof.G1_29DOF_JOINT_NAMES)
        assert sorted(ids) == list(range(29))
    # BFM-Zero gains follow kp = armature * (2*pi*10 Hz)^2 and kd = 2 * zeta * armature * (2*pi*10 Hz), zeta = 2
    _, _, kp = resolve_matching_names_values(actuator.stiffness, g1_29dof.G1_29DOF_JOINT_NAMES)
    _, _, kd = resolve_matching_names_values(actuator.damping, g1_29dof.G1_29DOF_JOINT_NAMES)
    _, _, armature = resolve_matching_names_values(actuator.armature, g1_29dof.G1_29DOF_JOINT_NAMES)
    omega = 2.0 * math.pi * 10.0
    for p, d, a in zip(kp, kd, armature):
        assert p == pytest.approx(a * omega**2, rel=1e-4)
        assert d == pytest.approx(4.0 * a * omega, rel=1e-4)


def test_g1_29dof_mirror_path_matches_isaaclab_layout(tmp_path):
    from isaaclab_neural.robots.g1_29dof import g1_29dof_mirror_path

    from isaaclab.utils import assets

    url = "https://example.org:8443/Assets/G1/configuration/base.usd"
    expected = assets._mirror_path(url, str(tmp_path))
    assert g1_29dof_mirror_path(url, str(tmp_path)) == expected
    assert g1_29dof_mirror_path("/local/file.usd", str(tmp_path)) == ""


def test_g1_29dof_root_layer_localization_rewrites_only_mirrored_urls(tmp_path):
    from isaaclab_neural.robots.g1_29dof import g1_29dof_mirror_path, localize_usd_root_layer

    from pxr import Sdf, UsdUtils

    mirrored_url = "https://example.org/Assets/G1/configuration/physics.usda"
    missing_url = "https://example.org/Assets/G1/configuration/missing.usda"
    payload = g1_29dof_mirror_path(mirrored_url, str(tmp_path))
    os.makedirs(os.path.dirname(payload))
    Sdf.Layer.CreateNew(payload).Save()
    root_path = str(tmp_path / "root.usda")
    root = Sdf.Layer.CreateNew(root_path)
    prim = Sdf.PrimSpec(root, "Robot", Sdf.SpecifierDef)
    prim.payloadList.Prepend(Sdf.Payload(mirrored_url))
    prim.referenceList.Prepend(Sdf.Reference(missing_url))
    root.Save()

    localized = localize_usd_root_layer(root_path, str(tmp_path))

    def asset_paths(path: str) -> set[str]:
        found: set[str] = set()
        UsdUtils.ModifyAssetPaths(Sdf.Layer.FindOrOpen(path), lambda p: (found.add(p), p)[1])
        return found

    assert localized != root_path
    assert asset_paths(localized) == {payload, missing_url}
    assert asset_paths(root_path) == {mirrored_url, missing_url}
    assert localize_usd_root_layer(localized, str(tmp_path)) == localized


def test_root_wrench_writes_only_free_joint_entries():
    from isaaclab_neural.envs.mdp_g1_29dof import write_root_wrench

    joint_f = torch.arange(18, dtype=torch.float32)
    starts = torch.tensor([0, 9])
    wrench = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]])
    write_root_wrench(joint_f, starts, wrench)
    torch.testing.assert_close(joint_f[0:6], wrench[0])
    torch.testing.assert_close(joint_f[9:15], wrench[1])
    torch.testing.assert_close(joint_f[6:9], torch.tensor([6.0, 7.0, 8.0]))
    torch.testing.assert_close(joint_f[15:18], torch.tensor([15.0, 16.0, 17.0]))


def test_g1_29dof_walking_task_uses_pr8055_gait_terms_with_bfm_actuators():
    from isaaclab_neural.controllers.bfm_zero import bfm_zero_action_scale
    from isaaclab_neural.envs.g1_29dof_env_cfg import G129DofFlatEnvCfg, G129DofPostureEnvCfg
    from isaaclab_neural.robots.g1_29dof import G1_29DOF_BFM_CFG, G1_29DOF_JOINT_NAMES, G1_29DOF_WALKING_JOINT_POS

    cfg = G129DofFlatEnvCfg()
    robot = cfg.scene.robot
    assert robot.actuators["body"].stiffness == G1_29DOF_BFM_CFG.actuators["body"].stiffness
    assert robot.soft_joint_pos_limit_factor == 0.9
    assert robot.init_state.joint_pos == G1_29DOF_WALKING_JOINT_POS
    assert list(cfg.actions.joint_pos.scale.values()) == bfm_zero_action_scale(G1_29DOF_JOINT_NAMES)
    for name in ("feet_air_time_variance", "feet_flight", "pelvis_height"):
        assert getattr(cfg.rewards, name).weight < 0.0
    assert cfg.terminations.base_height.params["minimum_height"] == 0.4
    posture = G129DofPostureEnvCfg()
    assert posture.rewards.pelvis_height is None


def test_g1_29dof_nerd_walking_task_matches_teacher_with_geometry_foot_contacts():
    import gymnasium as gym
    from isaaclab_neural.contacts.geometry_mdp import GeometryFeetAirTimePositiveBiped, GeometryFeetSlide
    from isaaclab_neural.envs.agents.g1_nerd_rsl_rl_ppo_cfg import G129DofFlatNeRDPPORunnerCfg
    from isaaclab_neural.envs.g1_29dof_env_cfg import (
        G1_29DOF_NERD_FEET_CONTACT_SEPARATION,
        G129DofFlatEnvCfg,
        NerdG129DofFlatEnvCfg,
    )
    from isaaclab_neural.envs.mdp_g1_29dof import GeometryFeetAirTimeVariance, geometry_feet_flight

    teacher, nerd = G129DofFlatEnvCfg(), NerdG129DofFlatEnvCfg()
    assert nerd.scene.contact_forces is None
    assert nerd.scene.robot.init_state == teacher.scene.robot.init_state
    assert nerd.actions.joint_pos.scale == teacher.actions.joint_pos.scale
    assert set(vars(nerd.rewards)) == set(vars(teacher.rewards))
    for name, term in vars(teacher.rewards).items():
        if term is not None:
            assert getattr(nerd.rewards, name).weight == term.weight, name
    geometry_terms = {
        "feet_air_time": GeometryFeetAirTimePositiveBiped,
        "feet_slide": GeometryFeetSlide,
        "feet_air_time_variance": GeometryFeetAirTimeVariance,
        "feet_flight": geometry_feet_flight,
    }
    for name, func in geometry_terms.items():
        term = getattr(nerd.rewards, name)
        assert term.func is func
        assert term.params["maximum_separation"] == G1_29DOF_NERD_FEET_CONTACT_SEPARATION
    assert nerd.terminations.base_height.params == teacher.terminations.base_height.params
    assert "maximum_separation" not in nerd.terminations.base_contact.params

    spec = gym.spec("Isaac-Velocity-Flat-G1-29DoF-NeRD-v0")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":G129DofFlatNeRDPPORunnerCfg")
    runner = G129DofFlatNeRDPPORunnerCfg()
    assert runner.actor.hidden_dims == runner.critic.hidden_dims == [512, 256, 128]
    assert runner.experiment_name == "g1_29dof_flat_nerd"


def test_g1_29dof_gait_terms_match_reference_calculations():
    from isaaclab_neural.envs import mdp_g1_29dof as terms

    air_time = torch.tensor([[0.0, 0.3], [0.2, 0.1], [0.0, 0.0]])
    data = SimpleNamespace(
        joint_pos=SimpleNamespace(torch=torch.tensor([[0.1, -0.2, 0.3]] * 3)),
        default_joint_pos=SimpleNamespace(torch=torch.zeros(3, 3)),
        root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.9], [0.0, 0.0, 0.3]])),
    )

    class Scene:
        sensors = {"feet": SimpleNamespace(data=SimpleNamespace(current_air_time=SimpleNamespace(torch=air_time)))}
        env_origins = torch.zeros(3, 3)

        def __getitem__(self, name):
            return SimpleNamespace(data=data)

    env = SimpleNamespace(scene=Scene(), num_envs=3, device="cpu", common_step_counter=0)
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 2])
    torch.testing.assert_close(
        terms.feet_flight(env, SimpleNamespace(name="feet", body_ids=[0, 1])), torch.tensor([0.0, 1.0, 0.0])
    )
    torch.testing.assert_close(terms.joint_deviation_l2(env, asset_cfg), torch.full((3,), 0.1**2 + 0.3**2))
    torch.testing.assert_close(
        terms.root_height_deficit_l2(env, 0.686, asset_cfg), torch.tensor([0.186**2, 0.0, 0.386**2])
    )
    assert not terms.root_height_below_minimum_after_warmup(env, 0.4, 10, asset_cfg).any()
    env.common_step_counter = 10
    torch.testing.assert_close(
        terms.root_height_below_minimum_after_warmup(env, 0.4, 10, asset_cfg), torch.tensor([False, False, True])
    )


def test_g1_29dof_air_time_variance_penalizes_asymmetric_completed_phases():
    from isaaclab_neural.envs.mdp_g1_29dof import feet_air_time_variance

    air_time = torch.zeros(1, 2)
    sensor = SimpleNamespace(data=SimpleNamespace(current_air_time=SimpleNamespace(torch=air_time)))
    command = torch.tensor([[0.5, 0.0, 0.0]])
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"feet": sensor}),
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    sensor_cfg = SimpleNamespace(name="feet", body_ids=[0, 1])
    cfg = SimpleNamespace(params={"sensor_cfg": sensor_cfg})
    term = feet_air_time_variance(cfg, env)

    def run(left_air: list[bool], right_air: list[bool]) -> torch.Tensor:
        penalty = torch.zeros(1)
        for left, right in zip(left_air, right_air):
            air_time[0] = torch.tensor([0.1 if left else 0.0, 0.1 if right else 0.0])
            penalty = term(env, "base_velocity", sensor_cfg)
        return penalty

    # both feet complete a 0.1 s swing and a 0.1 s stance phase: no variance
    pattern = [False, True, True, True, True, True, False, False, False, False, False, True]
    assert float(run(pattern, pattern)) == pytest.approx(0.0, abs=1e-9)
    term.reset()
    # the right foot swings for 0.2 s instead of 0.1 s
    left = [False, True, True, True, True, True, False, False, False, False, False, True, True]
    right = [False, True, True, True, True, True, True, True, True, True, True, False, False]
    assert float(run(left, right)) > 0.0


def test_g1_29dof_geometry_gait_terms_match_contact_sensor_terms(monkeypatch):
    from isaaclab_neural.envs import mdp_g1_29dof as terms

    contact = torch.zeros(3, 2, dtype=torch.bool)
    air_time = torch.zeros(3, 2)
    sensor = SimpleNamespace(data=SimpleNamespace(current_air_time=SimpleNamespace(torch=air_time)))
    command = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.3], [0.0, 0.4, 0.0]])
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"feet": sensor}),
        num_envs=3,
        device="cpu",
        step_dt=0.02,
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    sensor_cfg = SimpleNamespace(name="feet", body_ids=[0, 1])
    body_cfg = SimpleNamespace(name="robot", body_ids=[0, 1])
    separations = set()

    def geometry_contacts(body_ids, *, maximum_separation=0.0, env=None):
        separations.add(maximum_separation)
        return contact.clone()

    monkeypatch.setattr(terms, "body_contacts", geometry_contacts)
    sensor_term = terms.feet_air_time_variance(SimpleNamespace(params={"sensor_cfg": sensor_cfg}), env)
    geometry_term = terms.GeometryFeetAirTimeVariance(SimpleNamespace(params={"body_cfg": body_cfg}), env)
    generator = torch.Generator().manual_seed(0)
    total = torch.zeros(3)
    for _ in range(100):
        contact.copy_(torch.rand(3, 2, generator=generator) < 0.6)
        air_time.copy_(torch.where(contact, 0.0, air_time + env.step_dt))
        torch.testing.assert_close(
            terms.geometry_feet_flight(env, body_cfg, maximum_separation=0.003), terms.feet_flight(env, sensor_cfg)
        )
        expected = sensor_term(env, "base_velocity", sensor_cfg)
        torch.testing.assert_close(geometry_term(env, "base_velocity", body_cfg, maximum_separation=0.003), expected)
        total += expected
    assert separations == {0.003}
    assert total[0] > 0.0 and total[2] > 0.0 and total[1] == 0.0
