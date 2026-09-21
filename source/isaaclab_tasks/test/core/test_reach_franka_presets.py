# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch
from gymnasium.envs.registration import registry
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKPoseObjectiveCfg
from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.envs.mdp as mdp
from isaaclab.actuators import IdealPDActuatorCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import PresetCfg, resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_tasks.utils.preset_cli import enumerate_task_presets
from isaaclab_tasks.utils.preset_target import PresetTarget

from isaaclab_assets import FRANKA_PANDA_MENAGERIE_CFG

_TASK = "Isaac-Reach-Franka"
_OSC_TASK = "Isaac-Reach-Franka-OSC"
_CONTRIB_DIFFIK_ABS_TASK = "IsaacContrib-Reach-Franka-IK-Abs"


def _load_env_cfg(*presets: str):
    return _load_reach_env_cfg(_TASK, *presets)


def _load_reach_env_cfg(task: str, *presets: str):
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg = resolve_presets(cfg, selected=presets)
    assert cfg is not None
    return cfg


def _without_controller_dependent_cfg(cfg):
    cfg_dict = cfg.to_dict()
    cfg_dict.pop("actions")
    cfg_dict.pop("teleop_devices")
    for rigid_props in cfg_dict["scene"]["robot"]["spawn"]["rigid_props"]:
        rigid_props.pop("disable_gravity", None)
        rigid_props.pop("gravcomp", None)
    return cfg_dict


def test_reach_diffik_abs_legacy_task_is_a_deprecated_alias():
    spec = registry[_CONTRIB_DIFFIK_ABS_TASK]

    assert spec.kwargs["deprecated"] == {"alias": "--task Isaac-Reach-Franka physics=isaacsim_physx presets=diffik_abs"}
    with pytest.warns(FutureWarning, match="presets=diffik_abs"):
        legacy_cfg = load_cfg_from_registry(_CONTRIB_DIFFIK_ABS_TASK, "env_cfg_entry_point")

    canonical_cfg = _load_env_cfg("diffik_abs", "isaacsim_physx")
    legacy_cfg = resolve_presets(legacy_cfg)
    assert legacy_cfg is not None
    assert legacy_cfg.to_dict() == canonical_cfg.to_dict()


_REACH_PRESET_CASES = [
    (_TASK, (), "JointPositionActionCfg", "NewtonCfg"),
    (_TASK, ("isaacsim_physx",), "JointPositionActionCfg", "PhysxCfg"),
    (_TASK, ("newton_mjwarp",), "JointPositionActionCfg", "NewtonCfg"),
    (_TASK, ("ovphysx",), "JointPositionActionCfg", "OvPhysxCfg"),
    (_TASK, ("diffik",), "DifferentialInverseKinematicsActionCfg", "NewtonCfg"),
    (_TASK, ("diffik", "isaacsim_physx"), "DifferentialInverseKinematicsActionCfg", "PhysxCfg"),
    (_TASK, ("diffik", "newton_mjwarp"), "DifferentialInverseKinematicsActionCfg", "NewtonCfg"),
    (_TASK, ("diffik_abs", "isaacsim_physx"), "DifferentialInverseKinematicsActionCfg", "PhysxCfg"),
    (_TASK, ("diffik_abs", "newton_mjwarp"), "DifferentialInverseKinematicsActionCfg", "NewtonCfg"),
    (_TASK, ("diffik_abs", "ovphysx"), "DifferentialInverseKinematicsActionCfg", "OvPhysxCfg"),
    (_TASK, ("newton_ik", "newton_mjwarp"), "NewtonInverseKinematicsActionCfg", "NewtonCfg"),
    (_OSC_TASK, (), "OperationalSpaceControllerActionCfg", "NewtonCfg"),
    (_OSC_TASK, ("isaacsim_physx",), "OperationalSpaceControllerActionCfg", "PhysxCfg"),
    (_OSC_TASK, ("ovphysx",), "OperationalSpaceControllerActionCfg", "OvPhysxCfg"),
    ("Isaac-Reach-UR10", (), "JointPositionActionCfg", "NewtonCfg"),
    ("Isaac-Reach-UR10", ("isaacsim_physx",), "JointPositionActionCfg", "PhysxCfg"),
    ("Isaac-Reach-UR10", ("newton_mjwarp",), "JointPositionActionCfg", "NewtonCfg"),
]


@pytest.mark.parametrize(
    ("task", "presets", "action_type", "physics_type"),
    [pytest.param(*case, id=f"{case[0]}-{'-'.join(case[1]) or 'default'}") for case in _REACH_PRESET_CASES],
)
def test_reach_presets_resolve_supported_combinations(task, presets, action_type, physics_type):
    cfg = _load_reach_env_cfg(task, *presets)

    cfg.validate()
    assert type(cfg.actions.arm_action).__name__ == action_type
    assert type(cfg.sim.physics).__name__ == physics_type
    if task == _TASK:
        assert cfg.rewards.action_magnitude is None


def test_reach_ur10_physics_presets_change_only_physics():
    """UR10 backend selections must preserve the task configuration."""
    physx = _load_reach_env_cfg("Isaac-Reach-UR10", "isaacsim_physx")
    newton = _load_reach_env_cfg("Isaac-Reach-UR10", "newton_mjwarp")

    physx_cfg = physx.to_dict()
    newton_cfg = newton.to_dict()
    physx_cfg["sim"].pop("physics")
    newton_cfg["sim"].pop("physics")
    assert physx_cfg == newton_cfg
    assert newton.terminations.success.func is mdp.pose_command_success
    assert newton.rewards.success.weight == pytest.approx(10.0)
    assert newton.rewards.action_magnitude.weight == pytest.approx(-0.005)


def test_reach_action_presets_preserve_controller_independent_configuration():
    joint_pos_physx = _load_env_cfg("joint_pos", "isaacsim_physx")
    diffik_physx = _load_env_cfg("diffik", "isaacsim_physx")
    joint_pos_newton = _load_env_cfg("joint_pos", "newton_mjwarp")
    diffik_newton = _load_env_cfg("diffik", "newton_mjwarp")
    newton_ik = _load_env_cfg("newton_ik", "newton_mjwarp")

    assert _load_env_cfg().actions.arm_action.to_dict() == joint_pos_newton.actions.arm_action.to_dict()
    assert _without_controller_dependent_cfg(joint_pos_physx) == _without_controller_dependent_cfg(diffik_physx)
    assert _without_controller_dependent_cfg(joint_pos_newton) == _without_controller_dependent_cfg(diffik_newton)
    assert _without_controller_dependent_cfg(joint_pos_newton) == _without_controller_dependent_cfg(newton_ik)


@pytest.mark.parametrize(
    ("action_preset", "physics_preset"),
    [("diffik", "isaacsim_physx"), ("newton_ik", "newton_mjwarp")],
)
def test_reach_relative_ik_presets_configure_six_dof_native_teleop_devices(action_preset, physics_preset):
    cfg = _load_env_cfg(action_preset, physics_preset)

    cfg.validate()
    assert set(cfg.teleop_devices.devices) == {"keyboard", "gamepad", "spacemouse"}
    assert all(not device_cfg.gripper_term for device_cfg in cfg.teleop_devices.devices.values())


def test_reach_diffik_physx_configures_teleop_physics():
    cfg = _load_env_cfg("diffik", "isaacsim_physx")
    rigid_props = cfg.scene.robot.spawn.rigid_props

    physx_props = next(props for props in rigid_props if isinstance(props, PhysxRigidBodyCfg))
    assert physx_props.disable_gravity
    assert physx_props.max_depenetration_velocity == pytest.approx(5.0)
    assert not cfg.scene.robot.spawn.make_uninstanceable
    assert cfg.scene.robot.spawn.collision_props is None
    assert cfg.scene.robot.spawn.usd_path.endswith("/FrankaEmika/Legacy/panda_instanceable.usd")


def test_reach_newton_ik_configures_gravity_compensation():
    cfg = _load_env_cfg("newton_ik", "newton_mjwarp")
    rigid_props = cfg.scene.robot.spawn.rigid_props

    mujoco_props = next(props for props in rigid_props if isinstance(props, MujocoRigidBodyCfg))
    assert mujoco_props.gravcomp == pytest.approx(1.0)
    assert cfg.scene.robot.spawn.usd_path.endswith("/FrankaEmika/franka_panda.usda")


def test_reach_newton_ik_uses_native_se3_command_convention():
    cfg = _load_env_cfg("newton_ik", "newton_mjwarp")
    pose_objectives = [
        objective for objective in cfg.actions.arm_action.objectives if isinstance(objective, NewtonIKPoseObjectiveCfg)
    ]

    assert len(pose_objectives) == 1
    assert pose_objectives[0].command_type == "pose"
    assert pose_objectives[0].use_relative_mode
    assert len(pose_objectives[0].scale) == 6


def test_reach_default_preset_does_not_configure_se3_teleop_devices():
    cfg = _load_env_cfg()

    assert cfg.teleop_devices.devices == {}


def test_reach_success_requires_position_and_orientation():
    cfg = _load_env_cfg()

    assert cfg.commands.ee_pose.position_success_threshold == pytest.approx(0.05)
    assert cfg.commands.ee_pose.orientation_success_threshold == pytest.approx(0.2)
    assert cfg.terminations.success is None
    assert cfg.rewards.success is None
    assert cfg.terminations.time_out.func is mdp.time_out
    assert cfg.terminations.time_out.time_out is True
    fine_reward = cfg.rewards.end_effector_position_tracking_fine_grained
    assert fine_reward.func is mdp.position_command_error_tanh
    assert fine_reward.weight == pytest.approx(0.1)
    assert fine_reward.params["std"] == pytest.approx(0.1)
    assert fine_reward.params["asset_cfg"].body_names == ["panda_hand"]
    assert fine_reward.params["command_name"] == "ee_pose"
    assert cfg.rewards.action_magnitude is None
    assert cfg.rewards.action_rate.weight == pytest.approx(-0.0001)
    assert cfg.rewards.joint_vel.weight == pytest.approx(-0.0001)
    assert cfg.rewards.joint_vel.params["asset_cfg"].joint_names is None
    assert cfg.rewards.joint_vel.params["asset_cfg"].joint_ids == slice(None)
    assert cfg.curriculum.joint_vel.params == {"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}

    angles = torch.tensor([0.19, 0.19, 0.21])
    body_quaternions = torch.zeros(3, 1, 4)
    body_quaternions[:, 0, 2] = torch.sin(angles / 2)
    body_quaternions[:, 0, 3] = torch.cos(angles / 2)
    command_values = torch.zeros(3, 7)
    command_values[:, 6] = 1.0
    robot_data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=torch.zeros(3, 3)),
        root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)),
        body_pos_w=SimpleNamespace(torch=torch.tensor([[[0.04, 0.0, 0.0]], [[0.06, 0.0, 0.0]], [[0.04, 0.0, 0.0]]])),
        body_quat_w=SimpleNamespace(torch=body_quaternions),
        joint_vel=SimpleNamespace(torch=torch.tensor([[0.0] * 7 + [0.2, 0.3], [0.0] * 7 + [0.1, 0.2], [0.0] * 9])),
    )
    command = object.__new__(mdp.UniformPoseCommand)
    command.robot = SimpleNamespace(data=robot_data)
    command.body_idx = 0
    command.pose_command_b = command_values
    command.pose_command_w = torch.zeros_like(command_values)
    command.cfg = cfg.commands.ee_pose
    command._env = SimpleNamespace(num_envs=3, device=torch.device("cpu"))
    command._track_success = True
    command._succeeded = torch.zeros(3, dtype=torch.bool)

    class CommandManager:
        def get_term(self, name):
            assert name == "ee_pose"
            return command

        def get_command(self, name):
            assert name == "ee_pose"
            return command_values

    env = SimpleNamespace(
        command_manager=CommandManager(),
        scene={"robot": SimpleNamespace(data=robot_data)},
        episode_length_buf=torch.zeros(3, dtype=torch.long),
        max_episode_length=360,
    )
    torch.testing.assert_close(
        cfg.rewards.joint_vel.func(env, **cfg.rewards.joint_vel.params), torch.tensor([0.13, 0.05, 0.0])
    )
    fine_params = dict(fine_reward.params)
    fine_params["asset_cfg"] = SimpleNamespace(name="robot", body_ids=[0])
    torch.testing.assert_close(fine_reward.func(env, **fine_params), 1 - torch.tanh(torch.tensor([0.4, 0.6, 0.4])))
    assert not cfg.terminations.time_out.func(env).any()
    env.episode_length_buf[0] = env.max_episode_length
    assert torch.equal(cfg.terminations.time_out.func(env), torch.tensor([True, False, False]))
    succeeded = mdp.pose_command_success(env, command_name="ee_pose")

    assert torch.equal(succeeded, torch.tensor([True, False, False]))
    assert torch.equal(command._succeeded, succeeded)

    command.cfg = command.cfg.replace(orientation_success_threshold=None)
    command._succeeded.zero_()
    position_only_succeeded = mdp.pose_command_success(env, command_name="ee_pose")

    assert torch.equal(position_only_succeeded, torch.tensor([True, False, True]))


def test_reach_osc_effort_actuator_keeps_menagerie_velocity_limit():
    """The zero-gain effort actuator must keep the asset's solver velocity limit; the USD authors none."""
    cfg = _load_reach_env_cfg(_OSC_TASK)
    arm_actuator = cfg.scene.robot.actuators["panda_arm"]

    assert isinstance(arm_actuator, IdealPDActuatorCfg)
    assert arm_actuator.stiffness == 0.0 and arm_actuator.damping == 0.0
    assert arm_actuator.joint_velocity_limit == FRANKA_PANDA_MENAGERIE_CFG.actuators["panda_arm"].joint_velocity_limit


def test_reach_osc_resolves_controller_preset_values_to_defaults():
    """The OSC action term replaces the arm-controller presets, so their side effects must not leak in."""
    cfg = _load_reach_env_cfg(_OSC_TASK)
    cfg.validate()

    physx_props = next(props for props in cfg.scene.robot.spawn.rigid_props if isinstance(props, PhysxRigidBodyCfg))
    mujoco_props = next(props for props in cfg.scene.robot.spawn.rigid_props if isinstance(props, MujocoRigidBodyCfg))
    preset_map = enumerate_task_presets(_OSC_TASK)
    domain_presets = set(preset_map[PresetTarget.DOMAIN]) - set(preset_map[PresetTarget.PHYSICS])

    assert not isinstance(cfg.rewards.action_magnitude.weight, PresetCfg)
    assert cfg.rewards.action_magnitude.weight == _load_env_cfg().rewards.action_magnitude.weight
    assert physx_props.disable_gravity is True
    assert mujoco_props.gravcomp == pytest.approx(1.0)
    assert cfg.teleop_devices.devices == {}
    assert domain_presets == {"diffik_abs"}


def test_reach_osc_diffik_abs_is_a_deprecated_no_op_alias():
    """``presets=diffik_abs`` keeps resolving on the OSC task but warns and changes nothing."""
    default_cfg = _load_reach_env_cfg(_OSC_TASK, "ovphysx")
    default_cfg.validate()
    alias_cfg = _load_reach_env_cfg(_OSC_TASK, "diffik_abs", "ovphysx")

    with pytest.warns(FutureWarning, match="presets=diffik_abs"):
        alias_cfg.validate()

    assert type(alias_cfg.rewards.action_magnitude.weight) is float
    assert alias_cfg.to_dict() == default_cfg.to_dict()


def test_reach_newton_ik_rejects_physx():
    cfg = _load_env_cfg("newton_ik", "isaacsim_physx")

    with pytest.raises(ValueError, match="requires a Newton physics preset"):
        cfg.validate()
