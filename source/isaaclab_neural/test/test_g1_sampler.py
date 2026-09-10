# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Unitree G1 dataset sampler."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from isaaclab_neural.generate.samplers.g1 import G1TrajectorySampler


class _DummyAdapter:
    def __init__(self):
        self.num_envs = 2
        self.device = "cpu"
        self.joint_f_dim = 8
        self.actuator_writes = {}
        self.actuator_write_counts = {"stiffness": 0, "damping": 0}
        actuator = SimpleNamespace(
            stiffness=torch.tensor([[20.0, 200.0], [20.0, 200.0]]),
            damping=torch.tensor([[2.0, 5.0], [2.0, 5.0]]),
            joint_indices=torch.tensor([0, 1]),
        )
        data = SimpleNamespace(
            soft_joint_pos_limits=torch.tensor([[[-1.0, 1.0], [-2.0, 2.0]]]),
            soft_joint_vel_limits=torch.tensor([[10.0, 20.0]]),
            joint_effort_limits=torch.tensor([[30.0, 40.0]]),
        )

        def write_stiffness(**kwargs):
            self.actuator_writes["stiffness"] = kwargs
            self.actuator_write_counts["stiffness"] += 1

        def write_damping(**kwargs):
            self.actuator_writes["damping"] = kwargs
            self.actuator_write_counts["damping"] += 1

        robot = SimpleNamespace(
            data=data,
            actuators={"mixed": actuator},
            num_joints=2,
            write_joint_stiffness_to_sim_index=write_stiffness,
            write_joint_damping_to_sim_index=write_damping,
        )
        self.env = SimpleNamespace(scene={"robot": robot})
        self.state_bound_args = None
        self.force_bound_arg = None

    def action_bounds(self, low, high):
        return torch.tensor([low]), torch.tensor([high])

    def state_bounds(self, **kwargs):
        self.state_bound_args = kwargs
        return torch.zeros(17), torch.ones(17)

    def joint_force_bounds(self, value):
        self.force_bound_arg = value
        return value


def test_g1_sampler_uses_articulation_limits():
    adapter = _DummyAdapter()

    G1TrajectorySampler(adapter, trajectory_length=1, data_device="cpu")

    assert adapter.state_bound_args["joint_q_min"].shape == (9,)
    assert torch.equal(adapter.state_bound_args["joint_q_min"][-2:], torch.tensor([-1.0, -2.0]))
    assert torch.equal(adapter.state_bound_args["joint_q_max"][-2:], torch.tensor([1.0, 2.0]))
    assert torch.equal(adapter.state_bound_args["joint_qd_lim"][-2:], torch.tensor([10.0, 20.0]))
    assert torch.equal(adapter.force_bound_arg, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 40.0]))


def test_g1_sampler_spec_enables_relative_gain_randomization():
    from isaaclab_neural.generate.generate_dataset import _sampler_spec

    sampler_cls, kwargs = _sampler_spec(
        "Isaac-Velocity-Flat-G1-Dataset-Gen-v0",
        "G1-Native",
        "G1",
        True,
        (20.0, 80.0),
        (0.5, 4.0),
        (0.9, 1.1),
        (0.7, 1.3),
    )

    assert sampler_cls is G1TrajectorySampler
    assert kwargs == {
        "randomize_pd_gains": True,
        "kp_scale_range": (0.9, 1.1),
        "kd_scale_range": (0.7, 1.3),
        "hold_hand_actions": True,
    }


def test_g1_sampler_scales_nominal_gains_once_per_trajectory():
    adapter = _DummyAdapter()
    sampler = G1TrajectorySampler(
        adapter,
        trajectory_length=1,
        data_device="cpu",
        randomize_pd_gains=True,
        kp_scale_range=(0.5, 0.5),
        kd_scale_range=(2.0, 2.0),
    )

    sampler._pre_step_hook(0)

    actuator = adapter.env.scene["robot"].actuators["mixed"]
    assert torch.equal(actuator.stiffness, torch.tensor([[10.0, 100.0], [10.0, 100.0]]))
    assert torch.equal(actuator.damping, torch.tensor([[4.0, 10.0], [4.0, 10.0]]))
    assert torch.equal(adapter.actuator_writes["stiffness"]["stiffness"], actuator.stiffness)
    assert torch.equal(adapter.actuator_writes["damping"]["damping"], actuator.damping)

    sampler._pre_step_hook(1)

    assert adapter.actuator_write_counts == {"stiffness": 1, "damping": 1}
    assert torch.equal(actuator.stiffness, torch.tensor([[10.0, 100.0], [10.0, 100.0]]))
    assert torch.equal(actuator.damping, torch.tensor([[4.0, 10.0], [4.0, 10.0]]))


def test_g1_sampler_preserves_slice_joint_indices():
    adapter = _DummyAdapter()
    adapter.env.scene["robot"].actuators["mixed"].joint_indices = slice(None)
    sampler = G1TrajectorySampler(
        adapter,
        trajectory_length=1,
        data_device="cpu",
        randomize_pd_gains=True,
        kp_scale_range=(1.0, 1.0),
        kd_scale_range=(1.0, 1.0),
    )

    sampler._pre_step_hook(0)

    assert adapter.actuator_writes["stiffness"]["joint_ids"] == slice(None)
    assert adapter.actuator_writes["damping"]["joint_ids"] == slice(None)


def test_g1_sampler_writes_each_actuator_group_indices():
    adapter = _DummyAdapter()
    robot = adapter.env.scene["robot"]
    robot.num_joints = 4
    robot.actuators = {
        "legs": SimpleNamespace(
            stiffness=torch.tensor([[20.0, 30.0], [20.0, 30.0]]),
            damping=torch.tensor([[2.0, 3.0], [2.0, 3.0]]),
            joint_indices=torch.tensor([0, 1]),
        ),
        "arms": SimpleNamespace(
            stiffness=torch.tensor([[40.0], [40.0]]),
            damping=torch.tensor([[4.0], [4.0]]),
            joint_indices=torch.tensor([3]),
        ),
    }
    writes = []

    def write_stiffness(**kwargs):
        writes.append(kwargs["joint_ids"].clone())

    robot.write_joint_stiffness_to_sim_index = write_stiffness
    robot.write_joint_damping_to_sim_index = lambda **kwargs: None
    sampler = G1TrajectorySampler(
        adapter,
        trajectory_length=1,
        data_device="cpu",
        randomize_pd_gains=True,
        kp_scale_range=(1.0, 1.0),
        kd_scale_range=(1.0, 1.0),
    )

    sampler._pre_step_hook(0)

    assert len(writes) == 2
    assert torch.equal(writes[0], torch.tensor([0, 1]))
    assert torch.equal(writes[1], torch.tensor([3]))


def test_perturb_policy_action_adds_noise_and_clips():
    from isaaclab_neural.generate.sampler import perturb_policy_action

    action = torch.zeros((2, 3))
    low = torch.full((3,), -0.5)
    high = torch.full((3,), 0.5)

    assert torch.equal(perturb_policy_action(action, 0.0, low, high), action)

    torch.manual_seed(0)
    noisy = perturb_policy_action(action, 1.0, low, high, clip_to_bounds=True)
    assert noisy.shape == action.shape
    assert torch.all(noisy >= low)
    assert torch.all(noisy <= high)
    assert not torch.equal(noisy, action)


def test_perturb_policy_action_can_skip_clip():
    from isaaclab_neural.generate.sampler import perturb_policy_action

    action = torch.zeros((2, 3))
    low = torch.full((3,), -0.01)
    high = torch.full((3,), 0.01)

    torch.manual_seed(0)
    noisy = perturb_policy_action(action, 1.0, low, high, clip_to_bounds=False)
    assert noisy.shape == action.shape
    assert torch.any(noisy.abs() > high)


def test_action_hold_period_holds_across_control_decimation():
    from isaaclab_neural.generate.sampler import action_hold_period

    assert action_hold_period("frame", 4) == 4
    assert action_hold_period("frame", 1) == 1
    assert action_hold_period("env", 4) == 1


def test_action_hold_period_resamples_every_frame_for_pool_perturb():
    from isaaclab_neural.generate.sampler import action_hold_period

    assert action_hold_period("frame", 4, resample_actions_every_frame=True) == 1
    assert action_hold_period("env", 4, resample_actions_every_frame=True) == 1

    adapter = _DummyAdapter()
    adapter.env.cfg = SimpleNamespace(decimation=4)
    sampled = []
    processed = []

    def sample_action(low, high):
        action = torch.full((2, 1), float(len(sampled) + 1))
        sampled.append(action)
        return action

    adapter.sample_action = sample_action
    adapter.step_action_frame = lambda action, *, process_action: processed.append(process_action)
    sampler = G1TrajectorySampler(adapter, trajectory_length=6, data_device="cpu", hold_hand_actions=False)

    def collect(num_transitions, **kwargs):
        kwargs["pre_rollout_fn"]()
        return torch.stack([kwargs["step_fn"](step) for step in range(num_transitions)])

    sampler._sample_loop = collect
    actions = sampler.sample_trajectories_action_mode(6, resample_actions_every_frame=True, action_hold_frames=20)

    torch.testing.assert_close(actions[:, 0, 0], torch.arange(1, 7, dtype=torch.float32))
    assert processed == [True] * 6


def test_g1_hand_joint_indices_select_numbered_hand_joints():
    from isaaclab_neural.robots.g1 import g1_hand_joint_indices

    names = [
        "left_hip_pitch_joint",
        "left_one_joint",
        "right_six_joint",
        "torso_joint",
        "left_zero_joint",
    ]
    assert g1_hand_joint_indices(names) == [1, 2, 4]


def test_g1_action_mode_holds_hand_joints_at_reset_pose():
    adapter = _DummyAdapter()
    robot = adapter.env.scene["robot"]
    robot.data.joint_pos = torch.tensor([[0.10, 0.40], [0.10, 0.50]])
    robot.data.default_joint_pos = torch.zeros((2, 2))
    term = SimpleNamespace(
        _joint_names=["left_hip_pitch_joint", "left_one_joint"],
        _joint_ids=slice(None),
        _offset=robot.data.default_joint_pos.clone(),
        _scale=0.5,
    )
    adapter.env.action_manager = SimpleNamespace(get_term=lambda name: term)

    sampler = G1TrajectorySampler(adapter, trajectory_length=1, data_device="cpu")
    sampler._on_action_rollout_reset(constrain_sampled_actions=True)

    action = torch.ones((2, 2))
    held = sampler._constrain_sampled_action(action.clone())
    assert held[0, 0] == 1.0
    assert torch.allclose(held[:, 1], torch.tensor([0.8, 1.0]))


def test_g1_action_mode_can_disable_hand_hold():
    adapter = _DummyAdapter()
    robot = adapter.env.scene["robot"]
    robot.data.joint_pos = torch.tensor([[0.0, 0.4], [0.0, 0.4]])
    robot.data.default_joint_pos = torch.zeros((2, 2))
    term = SimpleNamespace(
        _joint_names=["left_hip_pitch_joint", "left_one_joint"],
        _joint_ids=slice(None),
        _offset=robot.data.default_joint_pos.clone(),
        _scale=0.5,
    )
    adapter.env.action_manager = SimpleNamespace(get_term=lambda name: term)

    sampler = G1TrajectorySampler(adapter, trajectory_length=1, data_device="cpu", hold_hand_actions=False)
    sampler._on_action_rollout_reset(constrain_sampled_actions=True)
    action = torch.ones((2, 2))
    assert torch.equal(sampler._constrain_sampled_action(action.clone()), action)
