# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for policy-action perturbation and multi-timescale action holds."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from isaaclab_neural.generate.arguments import get_parser
from isaaclab_neural.generate.sampler import ActionTrajectorySampler, perturb_policy_action


def test_perturb_policy_action_adds_noise_and_clips() -> None:
    action = torch.zeros((2, 3))
    low = torch.full((3,), -0.5)
    high = torch.full((3,), 0.5)

    assert torch.equal(perturb_policy_action(action, 0.0, low, high), action)

    torch.manual_seed(0)
    noisy = perturb_policy_action(action, 1.0, low, high, clip_to_bounds=True)
    assert noisy.shape == action.shape
    assert torch.all(noisy >= low) and torch.all(noisy <= high)
    assert not torch.equal(noisy, action)
    unclipped = perturb_policy_action(action, 1.0, low, high, clip_to_bounds=False)
    assert torch.any(unclipped.abs() > high)


class _RecordingAdapter:
    """Minimal adapter double that records when actions are (re)processed."""

    def __init__(self, num_frames: int):
        self.env = SimpleNamespace(cfg=SimpleNamespace(decimation=2))
        self.device = "cpu"
        self.num_envs = 1
        self.action_dim = 1
        self.processed_steps: list[int] = []
        self._num_frames = num_frames
        self._counter = 0

    def action_bounds(self, low: float = -1.0, high: float = 1.0):
        return torch.full((1,), low), torch.full((1,), high)

    def sample_action(self, low, high):
        self._counter += 1
        return torch.full((1, 1), float(self._counter))

    def zero_action(self):
        return torch.zeros((1, 1))

    def step_action_frame(self, action, *, process_action: bool = True):
        if process_action:
            self.processed_steps.append(self._step)
        return


def _run_hold_schedule(hold_frames, num_frames: int = 40) -> list[int]:
    adapter = _RecordingAdapter(num_frames)
    sampler = ActionTrajectorySampler.__new__(ActionTrajectorySampler)
    sampler.adapter = adapter
    sampler.action_low, sampler.action_high = adapter.action_bounds()

    def fake_sample_loop(num_transitions, *, step_fn, pre_rollout_fn=None, **_):
        pre_rollout_fn()
        for step in range(num_frames):
            adapter._step = step
            step_fn(step)
        return {}

    sampler._sample_loop = fake_sample_loop  # type: ignore[method-assign]
    sampler.sample_trajectories_action_mode(num_frames, action_hold_frames=hold_frames)
    return adapter.processed_steps


def test_single_hold_length_matches_fixed_period() -> None:
    assert _run_hold_schedule(20) == [0, 20]
    assert _run_hold_schedule([0]) == list(range(0, 40, 2))


def test_multiple_hold_lengths_are_sampled_per_segment() -> None:
    torch.manual_seed(3)
    steps = _run_hold_schedule([2, 10])
    gaps = {b - a for a, b in zip(steps, steps[1:])}
    assert steps[0] == 0
    assert gaps <= {2, 10}
    assert len(gaps) == 2


def test_empty_hold_choices_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one hold length"):
        _run_hold_schedule([])


def test_generation_cli_accepts_hold_list_noise_and_rl_contract() -> None:
    parser = get_parser()
    args, _ = parser.parse_known_args(
        ["--action-hold-frames", "2", "5", "40", "--policy-action-noise-std", "0.3", "--rl-action-contract"]
    )
    assert args.action_hold_frames == [2, 5, 40]
    assert args.policy_action_noise_std == pytest.approx(0.3)
    assert args.rl_action_contract is True
    assert parser.parse_known_args([])[0].action_hold_frames == [0]
