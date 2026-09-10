# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for action-hold resampling in dataset generation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from isaaclab_neural.generate.adapter import DataGenerationAdapter
from isaaclab_neural.generate.sampler import (
    policy_control_interval_dt,
    resolve_action_resample_period,
    should_refresh_policy_mdp,
)


def test_frame_hold_defaults_to_env_decimation() -> None:
    assert resolve_action_resample_period("frame", env_decimation=2, action_hold_frames=0) == 2
    assert resolve_action_resample_period("frame", env_decimation=4, action_hold_frames=0) == 4


def test_frame_hold_is_at_least_decimation() -> None:
    assert resolve_action_resample_period("frame", env_decimation=2, action_hold_frames=20) == 20
    assert resolve_action_resample_period("frame", env_decimation=4, action_hold_frames=2) == 4


def test_env_granularity_ignores_hold() -> None:
    assert resolve_action_resample_period("env", env_decimation=2, action_hold_frames=0) == 1
    assert resolve_action_resample_period("env", env_decimation=2, action_hold_frames=20) == 1


def test_policy_mdp_refreshes_after_each_control_interval() -> None:
    assert should_refresh_policy_mdp(0, decimation=4) is False
    assert should_refresh_policy_mdp(1, decimation=4) is False
    assert should_refresh_policy_mdp(3, decimation=4) is False
    assert should_refresh_policy_mdp(4, decimation=4) is True
    assert should_refresh_policy_mdp(8, decimation=4) is True


def test_policy_control_interval_prefers_step_dt() -> None:
    env = SimpleNamespace(step_dt=1.0 / 30.0, cfg=SimpleNamespace(sim=SimpleNamespace(dt=1.0 / 120.0)))
    assert policy_control_interval_dt(env, decimation=4) == pytest.approx(1.0 / 30.0)


def test_policy_control_interval_falls_back_to_sim_dt() -> None:
    env = SimpleNamespace(cfg=SimpleNamespace(sim=SimpleNamespace(dt=1.0 / 120.0)))
    assert policy_control_interval_dt(env, decimation=4) == pytest.approx(4.0 / 120.0)


def test_refresh_policy_mdp_updates_command_and_obs() -> None:
    class FakeCommandManager:
        def __init__(self) -> None:
            self.dts: list[float] = []

        def compute(self, dt: float) -> None:
            self.dts.append(dt)

    class FakeObservationManager:
        def compute(self, update_history: bool = False) -> dict[str, int]:
            return {"policy": int(update_history)}

    env = SimpleNamespace(
        command_manager=FakeCommandManager(),
        observation_manager=FakeObservationManager(),
        obs_buf=None,
    )
    adapter = DataGenerationAdapter.__new__(DataGenerationAdapter)
    adapter.env = env

    adapter.refresh_policy_mdp(1.0 / 30.0)

    assert env.command_manager.dts == pytest.approx([1.0 / 30.0])
    assert env.obs_buf == {"policy": 1}


def test_refresh_policy_mdp_skips_missing_managers() -> None:
    env = SimpleNamespace()
    adapter = DataGenerationAdapter.__new__(DataGenerationAdapter)
    adapter.env = env
    adapter.refresh_policy_mdp(0.1)


def test_policy_collection_refreshes_observations_before_each_control_tick() -> None:
    import torch
    from isaaclab_neural.generate.sampler import ActionTrajectorySampler

    decimation = 2
    frame = 0
    refresh_dts = []
    env = SimpleNamespace(cfg=SimpleNamespace(decimation=decimation), step_dt=0.2, obs_buf=torch.zeros(1, 1))

    def refresh(dt):
        refresh_dts.append(dt)
        env.obs_buf = torch.full((1, 1), float(frame))

    def step_action_frame(action, *, process_action):
        nonlocal frame
        frame += 1

    sampler = ActionTrajectorySampler.__new__(ActionTrajectorySampler)
    sampler.adapter = SimpleNamespace(env=env, refresh_policy_mdp=refresh, step_action_frame=step_action_frame)
    sampler.action_low = torch.tensor([-100.0])
    sampler.action_high = torch.tensor([100.0])

    def collect(num_transitions, **kwargs):
        kwargs["pre_rollout_fn"]()
        return torch.stack([kwargs["step_fn"](index).clone() for index in range(num_transitions)])

    sampler._sample_loop = collect
    actions = sampler.sample_trajectories_policy_mode(
        6, policy=lambda obs: obs, env_wrapped=SimpleNamespace(get_observations=lambda: env.obs_buf)
    )

    expected = (torch.arange(6) // decimation * decimation).float().reshape(6, 1, 1)
    torch.testing.assert_close(actions, expected)
    assert refresh_dts == pytest.approx([env.step_dt, env.step_dt])
